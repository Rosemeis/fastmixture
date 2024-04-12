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
parser.add_argument("-o", "--out", metavar="OUTPUT", default="fastmixture",
	help="Prefix output name (fastmixture)")
parser.add_argument("-s", "--seed", metavar="INT", type=int, default=42,
	help="Set random seed (42)")
parser.add_argument("-i", "--iter", metavar="INT", type=int, default=1000,
	help="Maximum number of iterations (1000)")
parser.add_argument("-e", "--tole", metavar="FLOAT", type=float, default=0.1,
	help="Tolerance in log-likelihood units between iterations (0.1)")
parser.add_argument("--levels", metavar="INT", type=int, default=5,
	help="Number of cyclic batch levels for powers of 2 (5)")
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


##### fastmixture #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print("-------------------------------------------------")
	print(f"fastmixture v0.5")
	print("C.G. Santander, A. Refoyo-Martinez and J. Meisner")
	print(f"K={args.K}, seed={args.seed}, levels={args.levels}, threads={args.threads}")
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
	lkVec = np.zeros(M)
	shared.loglike(G, P, Q, lkVec, args.threads)
	lkPre = np.sum(lkVec)
	print(f"Initial loglike: {round(lkPre,1)}\n")

	# Mini-batch parameters for stochastic EM
	print("Estimating Q and P using mini-batch EM.")
	check = 1
	batch = True
	batch_L = lkPre
	batch_N = [2**l for l in range(args.levels-1, -1, -1)]
	check_B = len(batch_N)
	print(f"Using {check_B} cyclic batch levels.")

	### EM algorithm
	a = np.zeros(N)
	Qa = np.zeros((N, args.K))
	Qb = np.zeros((N, args.K))

	# Prime iteration
	em.updateP(G, P, Q, Qa, Qb, a, args.threads)
	em.updateQ(Q, Qa, Qb, a)

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

	# fastmixture algorithm
	ts = time()
	np.random.seed(args.seed)
	for it in range(args.iter):
		if batch: # SQUAREM mini-batch updates
			B = batch_N[(check-1) % check_B]
			B_list = np.array_split(np.random.permutation(M), B)
			for b in np.arange(B):
				functions.squaremBatch(G, P, Q, a, P0, Q0, Qa, Qb, dP1, dP2, dP3, \
					dQ1, dQ2, dQ3, np.sort(B_list[b]), args.threads)
		else:
			# SQUAREM full update
			functions.squarem(G, P, Q, a, P0, Q0, Qa, Qb, dP1, dP2, dP3, \
				dQ1, dQ2, dQ3, args.threads)
		
		# Stabilization step
		em.updateP(G, P, Q, Qa, Qb, a, args.threads)
		em.updateQ(Q, Qa, Qb, a)

		# Log-likelihood convergence check
		if check % check_B == 0:
			shared.loglike(G, P, Q, lkVec, args.threads)
			lkCur = np.sum(lkVec)
			print(f"({it+1})\tLog-like: {round(lkCur,1)}\t" + \
				f"({round(time()-ts,1)}s)", flush=True)
			if batch:
				if (lkCur < batch_L) or (abs(lkCur - batch_L) < args.tole*check_B):
					batch_N = batch_N[1:]
					check_B = len(batch_N)
					if check_B > 1:
						print(f"Using {check_B} cyclic batch levels.")
						check = 0
					else:
						batch = False
						print("Running non-cyclic SQUAREM updates.")
				else:
					batch_L = lkCur
			else:
				if (abs(lkCur - lkPre) < args.tole):
					print("Converged!")
					print(f"Final log-likelihood: {round(lkCur,1)}")
					converged = True
					break
			lkPre = lkCur
			ts = time()
		check += 1
	
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
		log.write(f"\nFinal log-likelihood: {round(lkCur,1)}\n")
		if converged:
			log.write(f"Converged in {it+1} iterations.\n")
		else:
			log.write("EM algorithm did not converge in {args.iter} iterations!\n")
		log.write(f"Total elapsed time: {t_min}m{t_sec}s\n")
		log.write(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q\n")
		if not args.no_freqs:
			log.write(f"Saved P matrix as {args.out}.K{args.K}.s{args.seed}.P\n")
