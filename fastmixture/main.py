"""
fastmixture.
Main caller. Ancestry estimation.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os
import sys
from datetime import datetime
from time import time

### Argparse
parser = argparse.ArgumentParser(prog="fastmixture")
parser.add_argument("--version", action="version",
	version="%(prog)s v0.5")
parser.add_argument("-b", "--bfile", metavar="PLINK",
	help="Prefix for PLINK files (.bed, .bim, .fam)")
parser.add_argument("-k", "--K", metavar="INT", type=int,
	help="Number of ancestral components")
parser.add_argument("-t", "--threads", metavar="INT", type=int, default=1,
	help="Number of threads (1)")
parser.add_argument("-s", "--seed", metavar="INT", type=int, default=42,
	help="Set random seed (42)")
parser.add_argument("-o", "--out", metavar="OUTPUT", default="fastmixture",
	help="Prefix output name (fastmixture)")
parser.add_argument("--iter", metavar="INT", type=int, default=1000,
	help="Maximum number of iterations (1000)")
parser.add_argument("--tole", metavar="FLOAT", type=float, default=0.5,
	help="Tolerance in log-likelihood units between iterations (0.5)")
parser.add_argument("--q-tole", metavar="FLOAT", type=float, default=1e-6,
	help="Tolerance in RMSE for Q between iterations (1e-6)")
parser.add_argument("--batches", metavar="INT", type=int, default=32,
	help="Number of maximum mini-batches (32)")
parser.add_argument("--check", metavar="INT", type=int, default=5,
	help="Number of iterations between check for convergence")
parser.add_argument("--power", metavar="INT", type=int, default=11,
	help="Number of power iterations in randomized SVD (11)")
parser.add_argument("--svd-batch", metavar="INT", type=int, default=8192,
	help="Number of SNPs in SVD batches (8192)")
parser.add_argument("--als-iter", metavar="INT", type=int, default=1000,
	help="Maximum number of iterations in ALS (1000)")
parser.add_argument("--als-tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for RMSE of P between iterations (1e-4)")
parser.add_argument("--no-freqs", action="store_true",
	help="Do not save P-matrix")
parser.add_argument("--prime", metavar="INT", type=int, default=3,
	help="Number of priming iterations (3)")


##### fastmixture #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print("-------------------------------------------------")
	print(f"fastmixture v0.5")
	print("C.G. Santander, A. Refoyo-Martinez and J. Meisner")
	print(f"K={args.K}, seed={args.seed}, batches={args.batches}, threads={args.threads}")
	print("-------------------------------------------------\n")
	assert args.bfile is not None, "No input data (--bfile)!"
	assert args.K > 1, "Please set K > 1 (--K)!"
	start = time()

	# Create log-file of arguments
	full = vars(parser.parse_args())
	deaf = vars(parser.parse_args([]))
	mand = ["seed", "batches"]
	with open(f"{args.out}.K{args.K}.s{args.seed}.log", "w") as log:
		log.write("fastmixture v0.5\n")
		log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
		log.write(f"Directory: {os.getcwd()}\n")
		log.write("Options:\n")
		for key in full:
			if full[key] != deaf[key]:
				if type(full[key]) is bool:
					log.write(f"\t--{key}\n")
				else:
					log.write(f"\t--{key} {full[key]}\n")
			elif key in mand:
				log.write(f"\t--{key} {full[key]}\n")
	del full, deaf, mand

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

	# Load numerical libraries
	import numpy as np
	from math import ceil, log
	from fastmixture import em
	from fastmixture import em_batch
	from fastmixture import functions
	from fastmixture import shared

	### Read data
	# Finding length of .fam and .bim file
	assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
	print("Reading data...", end="", flush=True)
	N = functions.extract_length(f"{args.bfile}.fam")
	M = functions.extract_length(f"{args.bfile}.bim")
	G = np.zeros((M, N), dtype=np.uint8)

	# Read .bed file
	with open(f"{args.bfile}.bed", "rb") as bed:
		B = np.fromfile(bed, dtype=np.uint8, offset=3)
	N_bytes = ceil(N/4) # Length of bytes to describe N individuals
	shared.expandGeno(B, G, N_bytes, args.threads)
	del B
	print(f"\rLoaded {N} samples and {M} SNPs.")

	# Initalize parameters
	f = np.zeros(M)
	shared.estimateFreq(G, f, args.threads)

	# Initialize P and Q matrices from SVD and ALS
	ts = time()
	print("Performing SVD and ALS.", end="", flush=True)
	U, V = functions.randomizedSVD(G, f, args.K-1, args.svd_batch, \
		args.power, args.seed, args.threads)
	P, Q = functions.extractFactor(U, V, f, args.K, args.als_iter, args.als_tole, \
		args.seed)
	print(f"\rExtracted factor matrices ({round(time()-ts,1)} seconds).")
	del f, U, V

	# Estimate initial log-likelihood
	ts = time()
	l_vec = np.zeros(M)
	shared.loglike(G, P, Q, l_vec, args.threads)
	L_pre = np.sum(l_vec)
	print(f"Initial loglike: {round(L_pre,1)}\n")

	# Mini-batch parameters for stochastic EM
	batch = True
	batch_L = L_pre

	### EM algorithm
	Q_new = np.zeros((N, args.K))

	# Setup containers for EM algorithm
	converged = False
	P0 = np.zeros((M, args.K))
	Q0 = np.zeros((N, args.K))
	dP1 = np.zeros((M, args.K))
	dP2 = np.zeros((M, args.K))
	dP3 = np.zeros((M, args.K))
	dQ1 = np.zeros((N, args.K))
	dQ2 = np.zeros((N, args.K))
	dQ3 = np.zeros((N, args.K))

	# Accelerated priming iteration
	ts = time()
	em.updateP(G, P, Q, Q_new, args.threads)
	em.updateQ(Q, Q_new, M, args.threads)
	functions.squarem(G, P, Q, P0, Q0, Q_new, dP1, dP2, dP3, dQ1, dQ2, dQ3, \
		args.threads)
	em.updateP(G, P, Q, Q_new, args.threads)
	em.updateQ(Q, Q_new, M, args.threads)
	print(f"Performed priming iteration\t({round(time()-ts,1)}s)\n", flush=True)

	# fastmixture algorithm
	ts = time()
	print("Estimating Q and P using mini-batch EM.")
	print(f"Using {args.batches} mini-batches.")
	np.random.seed(args.seed)
	for it in range(args.iter):
		if batch: # SQUAREM mini-batch updates
			B_list = np.array_split(np.random.permutation(M), args.batches)
			for b in B_list:
				b_array = np.sort(b)
				functions.squaremBatch(G, P, Q, P0, Q0, Q_new, dP1, dP2, dP3, \
					dQ1, dQ2, dQ3, b_array, args.threads)
		
			# Stabilization step
			em.updateP(G, P, Q, Q_new, args.threads)
			em.updateQ(Q, Q_new, M, args.threads)

		# SQUAREM full update
		functions.squarem(G, P, Q, P0, Q0, Q_new, dP1, dP2, dP3, \
			dQ1, dQ2, dQ3, args.threads)
		
		# Stabilization step
		em.updateP(G, P, Q, Q_new, args.threads)
		em.updateQ(Q, Q_new, M, args.threads)

		# Log-likelihood convergence check
		if (it + 1) % args.check == 0:
			shared.loglike(G, P, Q, l_vec, args.threads)
			L_cur = np.sum(l_vec)
			print(f"({it+1})\tLog-like: {round(L_cur,1)}\t" + \
				f"({round(time()-ts,1)}s)", flush=True)
			if batch:
				if (L_cur < batch_L) or (abs(L_cur - batch_L) < args.tole):
					batch_L = float('-inf')
					args.batches = args.batches//2
					if args.batches > 1:
						print(f"Using {args.batches} mini-batches.")
					else:
						print("Running standard SQUAREM updates.")
						del B_list
						batch = False
						L_pre = float('-inf')
						Q_pre = np.copy(Q)
				else:
					batch_L = L_cur
			else:
				rmseQ = shared.rmse(Q, Q_pre)
				if (abs(L_cur - L_pre) < args.tole) or (rmseQ < args.q_tole):
					print("Converged!")
					print(f"Final log-likelihood: {round(L_cur,1)}")
					converged = True
					break
				np.copyto(Q_pre, Q, casting="no")
				L_pre = L_cur
			ts = time()

	# Print elapsed time for estimation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")

	### Save estimates and write output to log-file
	np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.Q", Q, fmt="%.6f")
	if not args.no_freqs:
		np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.P", P, fmt="%.6f")
	with open(f"{args.out}.K{args.K}.s{args.seed}.log", "a") as log:
		log.write(f"\nFinal log-likelihood: {round(L_cur,1)}\n")
		if converged:
			log.write(f"Converged in {it+1} iterations.\n")
		else:
			log.write("EM algorithm did not converge in {args.iter} iterations!\n")
		log.write(f"Total elapsed time: {t_min}m{t_sec}s\n")
		log.write(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q\n")
		if not args.no_freqs:
			log.write(f"Saved P matrix as {args.out}.K{args.K}.s{args.seed}.P\n")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'fastmixture' command!"
