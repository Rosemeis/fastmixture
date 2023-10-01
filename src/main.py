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
	version="%(prog)s v0.3")
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
parser.add_argument("-e", "--tole", metavar="FLOAT", type=float, default=1.0,
	help="Tolerance in log-likelihood units between c-th iterations (1.0)")
parser.add_argument("-c", "--check", metavar="INT", type=int, default=10,
	help="Iteration to estimate log-likelihood for convergence check (10)")
parser.add_argument("--num_batches", metavar="INT", type=int, default=32,
	help="Number of mini-batches (32)")
parser.add_argument("--power", metavar="INT", type=int, default=11,
	help="Number of power iterations in randomized SVD (11)")
parser.add_argument("--svd_batch", metavar="INT", type=int, default=4096,
	help="Number of SNPs in SVD batches (4096)")
parser.add_argument("--als_iter", metavar="INT", type=int, default=1000,
	help="Maximum number of iterations in ALS (1000)")
parser.add_argument("--als_tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for RMSE of P between iterations (1e-5)")
parser.add_argument("--als_save", action="store_true",
	help="Save initialized factor matrices from ALS")
parser.add_argument("--no_freqs", action="store_true",
	help="Do not save P-matrix")
parser.add_argument("--no_batch", action="store_true",
	help="Turn off mini-batch updates")
parser.add_argument("--verbose", action="store_true",
	help="DEBUG: More detailed output for debugging.")


##### fastmixture #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print("-------------------------------------------------")
	print(f"fastmixture v0.3")
	print("C.G. Santander, A. Refoyo-Martinez and J. Meisner")
	print(f"Parameters: K={args.K}, seed={args.seed}, threads={args.threads}")
	print("-------------------------------------------------\n")
	assert args.bfile is not None, "No input data (--bfile)!"
	assert args.K > 1, "Please set K > 1 (--K)!"
	assert (args.num_batches & (args.num_batches-1) == 0) and args.num_batches != 0, \
		"Please set number of batches to a power of 2 (--num_batches)!"
	start = time()

	# Create log-file of arguments
	full = vars(parser.parse_args())
	deaf = vars(parser.parse_args([]))
	mand = ["seed", "num_batches"]
	with open(f"{args.out}.K{args.K}.s{args.seed}.log", "w") as log:
		log.write("fastmixture v0.3\n")
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
	from math import ceil
	from src import em
	from src import functions
	from src import shared

	### Read data
	print("Reading data...", end="\r")
	# Finding length of .fam and .bim file
	assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
	N = functions.extract_length(f"{args.bfile}.fam")
	M = functions.extract_length(f"{args.bfile}.bim")

	# Read .bed file
	with open(f"{args.bfile}.bed", "rb") as bed:
		G = np.fromfile(bed, dtype=np.uint8, offset=3)
	B = ceil(N/4) # Length of bytes to describe n individuals
	G.shape = (M, B)
	print(f"Loaded {N} samples and {M} SNPs.", flush=True)

	# Initalize parameters
	f = np.zeros(M, dtype=np.float32)
	shared.estimateFreq(G, f, N, args.threads)

	# Initialize P and Q matrices from SVD and ALS
	ts = time()
	print("Initializing P and Q.", end="\r")
	U, V = functions.randomizedSVD(G, f, N, args.K-1, args.svd_batch, \
		args.power, args.seed, args.threads)
	if args.verbose:
		print("Performed Randomized SVD.")
	P, Q = functions.extractFactor(U, V, f, args.K, args.als_iter, args.als_tole, \
		args.seed, args.verbose)
	print(f"Extracted factor matrices ({round(time()-ts,1)} seconds).")
	del f, U, V

	# Optional save of initial factor matrices (debug feature)
	if args.als_save:
		np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.als.Q", Q, fmt="%.6f")
		if not args.no_freqs:
			np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.als.P", P, fmt="%.6f")

	# Mini-batch parameters for stochastic EM
	if args.no_batch:
		batch = False
	else:
		batch = True
		batch_N = args.num_batches

	# Estimate initial log-likelihood
	ts = time()
	lkVec = np.zeros(M)
	shared.loglike(G, P, Q, lkVec, args.threads)
	lkPre = np.sum(lkVec)
	print(f"Initial loglike: {round(lkPre,1)}", flush=True)

	### Setup containers for EM algorithm
	converged = False
	a = np.zeros(N, dtype=np.float32)
	pr = np.zeros(M, dtype=np.float32)
	pv = np.zeros(M, dtype=np.float32)
	P0 = np.zeros((M, args.K), dtype=np.float32)
	Pa = np.zeros((M, args.K), dtype=np.float32)
	Pb = np.zeros((M, args.K), dtype=np.float32)
	Q0 = np.zeros((N, args.K), dtype=np.float32)
	Qa = np.zeros((N, args.K), dtype=np.float32)
	Qb = np.zeros((N, args.K), dtype=np.float32)
	dP1 = np.zeros((M, args.K), dtype=np.float32)
	dP2 = np.zeros((M, args.K), dtype=np.float32)
	dP3 = np.zeros((M, args.K), dtype=np.float32)
	dQ1 = np.zeros((N, args.K), dtype=np.float32)
	dQ2 = np.zeros((N, args.K), dtype=np.float32)
	dQ3 = np.zeros((N, args.K), dtype=np.float32)
	if not args.no_batch:
		print("Estimating Q and P using mini-batch EM.")
		print(f"Using {batch_N} mini-batches.")
	else:
		print("Estimating Q and P using EM.")

	### EM algorithm
	ts = time()
	np.random.seed(args.seed)
	for it in range(1, args.iter+1):
		if batch: # SQUAREM mini-batch updates
			B_list = np.array_split(np.random.permutation(M), batch_N)
			for b in B_list:
				Bs = np.sort(b)
				functions.squaremBatch(G, P, Q, a, pr, pv, P0, Pa, Pb, Q0, Qa, \
					Qb, dP1, dP2, dP3, dQ1, dQ2, dQ3, \
					Bs, args.threads)
		else: # SQUAREM full updates
			functions.squarem(G, P, Q, a, pr, pv, P0, Pa, Pb, Q0, Qa, Qb, \
				dP1, dP2, dP3, dQ1, dQ2, dQ3, args.threads)
		
		# SQUAREM stabilization step
		em.updateP(G, P, Q, Pa, Pb, Qa, Qb, a, args.threads)
		em.updateQ(Q, Qa, Qb, a)

		# Log-likelihood convergence check
		if it % args.check == 0:
			shared.loglike(G, P, Q, lkVec, args.threads)
			lkCur = np.sum(lkVec)
			print(f"Iteration {it},\tLog-like: {round(lkCur,1)}\t " + \
				f"({round(time()-ts,1)} seconds)", flush=True)
			if batch:
				if (lkCur < lkPre) or (abs(lkCur - lkPre) < args.tole):
					batch_N = batch_N//2
					if batch_N > 1:
						print(f"Using {batch_N} mini-batches.")
					else:
						batch = False
						print("Running standard SQUAREM updates.")
			else:
				if (abs(lkCur - lkPre) < args.tole):
					print("Converged!")
					print(f"Final log-likelihood: {round(lkCur,1)}")
					converged = True
					break
			lkPre = lkCur
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
		log.write(f"\nFinal log-likelihood: {round(lkCur,1)}\n")
		if converged:
			log.write(f"Converged in {it} iterations.\n")
		else:
			log.write("EM algorithm did not converge!\n")
		log.write(f"Total elapsed time: {t_min}m{t_sec}s\n")
		log.write(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q\n")
		if not args.no_freqs:
			log.write(f"Saved P matrix as {args.out}.K{args.K}.s{args.seed}.P\n")
