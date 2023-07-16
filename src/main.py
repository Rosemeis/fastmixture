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
	version="%(prog)s v0.1")
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
parser.add_argument("-e", "--tole", metavar="FLOAT", type=float, default=0.5,
	help="Tolerance in log-likelihood units between c-th iterations (0.5)")
parser.add_argument("-c", "--check", metavar="INT", type=int, default=10,
	help="Iteration to estimate log-likelihood for convergence check (10)")
parser.add_argument("--num_batches", metavar="INT", type=int, default=16,
	help="Number of mini-batches (16)")
parser.add_argument("--power", metavar="INT", type=int, default=7,
	help="Number of power iterations in randomized SVD (7)")
parser.add_argument("--als_iter", metavar="INT", type=int, default=500,
	help="Maximum number of iterations in NMF (500)")
parser.add_argument("--als_tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for RMSE of P between iterations (1e-5)")
parser.add_argument("--no_freqs", action="store_true",
	help="Do not save P-matrix")
parser.add_argument("--als_save", action="store_true",
	help="DEBUG: Save initialized factor matrices")
parser.add_argument("--no_batch", action="store_true",
	help="DEBUG: Turn off mini-batch updates")
parser.add_argument("--verbose", action="store_true",
	help="DEBUG: More detailed output for debugging.")


##### fastmixture #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print(f"fastmixture v0.1")
	print("C.G. Santander, A. Refoyo-Martinez and J. Meisner")
	print("Parameters: K={}, seed={}, num_batches={}, threads={}\n".format(args.K, \
		args.seed, args.num_batches, args.threads))
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
		log.write("fastmixture v0.1\n")
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
	from src import em_batch
	from src import functions

	### Read data
	print("Reading data...", end="\r")
	# Finding length of .fam and .bim file
	N = functions.extract_length(f"{args.bfile}.fam")
	M = functions.extract_length(f"{args.bfile}.bim")

	# Read .bed file
	with open(f"{args.bfile}.bed", "rb") as bed:
		G = np.fromfile(bed, dtype=np.uint8, offset=3)
	B = ceil(N/4) # Length of bytes to describe n individuals
	G.shape = (M, B)
	print(f"Loaded {N} samples and {M} SNPs.", flush=True)

	### Initalize parameters
	converged = False
	f = np.zeros(M)
	a = np.zeros(N)
	lkVec = np.zeros(M)
	sumP1 = np.zeros(M) # Helper vector
	sumP2 = np.zeros(M) # Helper vector
	em.estimateFreq(G, f, N, args.threads)

	# Initialize P and Q matrices from SVD and ALS
	ts = time()
	print("Initializing P and Q.", end="\r")
	U, V = functions.randomizedSVD(G, f, N, args.K-1, args.num_batches, args.power, \
		args.seed, args.threads, args.verbose)
	P, Q = functions.extractFactor(U, V, f, args.K, args.als_iter, args.als_tole, \
		args.seed, args.verbose)
	print(f"Extracted factor matrices ({round(time()-ts,1)} seconds).")
	del U, V

	# Optional save of initial factor matrices (debug feature)
	if args.als_save:
		np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.als.Q", Q, fmt="%.5f")
		if not args.no_freqs:
			np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.als.P", P, fmt="%.5f")

	# Mini-batch parameters for stochastic EM
	if args.no_batch:
		batch = False
	else:
		batch = True
		batch_N = args.num_batches

	# Estimate allele frequencies and initial log-likelihood
	ts = time()
	em.loglike(G, P, Q, lkVec, args.threads)
	lkPre = np.sum(lkVec)
	print(f"Initial loglike: {round(lkPre,1)}", flush=True)

	### Setup matrices for EM algorithm
	sumQA = np.zeros((N, args.K))
	sumQB = np.zeros((N, args.K))
	diffP1 = np.zeros((M, args.K))
	diffP2 = np.zeros((M, args.K))
	diffP3 = np.zeros((M, args.K))
	diffQ1 = np.zeros((N, args.K))
	diffQ2 = np.zeros((N, args.K))
	diffQ3 = np.zeros((N, args.K))
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
				functions.squaremBatch(G, P, Q, a, sumP1, sumP2, sumQA, sumQB, \
					diffP1, diffP2, diffP3, diffQ1, diffQ2, diffQ3, Bs, batch_N, \
					args.threads)
		else: # SQUAREM full updates
			functions.squarem(G, P, Q, a, sumP1, sumP2, sumQA, sumQB, \
				diffP1, diffP2, diffP3, diffQ1, diffQ2, diffQ3, args.threads)
		
		# SQUAREM stabilization step
		em.updateP(G, P, Q, sumQA, sumQB, a, args.threads)
		em.updateQ(Q, sumQA, sumQB, a)

		# Convergence check
		if it % args.check == 0:
			em.loglike(G, P, Q, lkVec, args.threads)
			lkCur = np.sum(lkVec)
			print("Iteration {},\tLog-like: {}\t({} seconds)".format(
				it,round(lkCur,1),round(time()-ts,1)), flush=True)
			if batch:
				if (lkCur < lkPre) or (abs(lkCur - lkPre) < args.tole):
					batch_N = batch_N // 2
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

	### Save estimates and write output to log-file
	np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.Q", Q, fmt="%.5f")
	if not args.no_freqs:
		np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.P", P, fmt="%.5f")
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")
	with open(f"{args.out}.K{args.K}.s{args.seed}.log", "a") as log:
		log.write(f"\nFinal log-likelihood: {round(lkCur,1)}\n")
		if not converged:
			log.write("EM algorithm did not converge!\n")
		log.write(f"Total elapsed time: {t_min}m{t_sec}s\n")
		log.write(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q\n")
		if not args.no_freqs:
			log.write(f"Saved P matrix as {args.out}.K{args.K}.s{args.seed}.P\n")
