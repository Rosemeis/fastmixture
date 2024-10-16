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
	version="%(prog)s v0.93.1")
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
parser.add_argument("--batches", metavar="INT", type=int, default=32,
	help="Number of initial mini-batches (32)")
parser.add_argument("--supervised", metavar="FILE",
	help="Path to population assignment file")
parser.add_argument("--check", metavar="INT", type=int, default=5,
	help="Number of iterations between check for convergence")
parser.add_argument("--power", metavar="INT", type=int, default=12,
	help="Number of power iterations in randomized SVD (12)")
parser.add_argument("--chunk", metavar="INT", type=int, default=8192,
	help="Number of SNPs in chunk operations (8192)")
parser.add_argument("--als-iter", metavar="INT", type=int, default=1000,
	help="Maximum number of iterations in ALS (1000)")
parser.add_argument("--als-tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for RMSE of P between iterations (1e-4)")
parser.add_argument("--no-freqs", action="store_true",
	help="Do not save P-matrix")
parser.add_argument("--random-init", action="store_true",
	help="Random initialization of parameters")
parser.add_argument("--safety", action="store_true",
	help="Add extra safety steps in unstable optimizations")



##### fastmixture #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print("-------------------------------------------------")
	print(f"fastmixture v0.93.1")
	print("C.G. Santander, A. Refoyo-Martinez and J. Meisner")
	print(f"K={args.K}, seed={args.seed}, batches={args.batches}, threads={args.threads}")
	print("-------------------------------------------------\n")
	assert args.bfile is not None, "No input data (--bfile)!"
	assert args.K > 1, "Please set K > 1 (--K)!"
	assert args.batches > 1, "Please set a valid number of batches (--batches)!"
	start = time()

	# Create log-file of arguments
	full = vars(parser.parse_args())
	deaf = vars(parser.parse_args([]))
	mand = ["seed", "batches"]
	with open(f"{args.out}.K{args.K}.s{args.seed}.log", "w") as log:
		log.write("fastmixture v0.93.1\n")
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
	from fastmixture import functions
	from fastmixture import shared

	### Read data
	assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
	print("Reading data...", end="", flush=True)
	G, M, N = functions.readPlink(args.bfile, args.threads)
	print(f"\rLoaded {N} samples and {M} SNPs.")

	# Supervised setting
	if args.supervised is not None:
		print("Ancestry estimation in supervised mode!")
		y = np.loadtxt(args.supervised, dtype=np.uint8).reshape(-1)
		assert y.shape[0] == N, "Number of samples differ between files!"
		assert np.max(y) <= args.K, "Wrong number of ancestral sources!"
		assert np.min(y) >= 0, "Wrong format in population assignments!"
		print(f"{np.sum(y > 0)}/{N} individuals with fixed ancestry.")

		# Count individuals in ancestral sources
		z, x = np.unique(y[y > 0], return_counts=True)
		z -= 1
		x = x[np.argsort(z)]
		z = np.sort(z)

		# Setup containers and initialize
		np.random.seed(args.seed) # Set random seed
		P = np.random.rand(M, args.K)
		Q = np.random.rand(N, args.K)
		P[:,z] = 0.0
		shared.initP(G, P, y, x, args.threads)
		shared.initQ(Q, y)
		del z, x
	else:
		# Initalize parameters in unsupervised mode
		y = None

		# Random initialization
		if args.random_init:
			print("Random initialization.")
			np.random.seed(args.seed) # Set random seed
			P = np.random.rand(M, args.K)
			P.clip(min=1e-5, max=1-(1e-5), out=P)
			Q = np.random.rand(N, args.K)
			Q.clip(min=1e-5, max=1-(1e-5), out=Q)
			Q /= np.sum(Q, axis=1, keepdims=True)
		else: # SVD-based initialization
			f = np.zeros(M)
			shared.estimateFreq(G, f, args.threads)
			assert (np.min(f) > 0.0) & (np.max(f) < 1.0), "Please perform MAF filtering!"

			# Initialize P and Q matrices from SVD and ALS
			ts = time()
			print("Performing SVD and ALS.", end="", flush=True)
			U, V = functions.randomizedSVD(G, f, args.K-1, args.chunk, args.power, \
				args.seed, args.threads)
			P, Q = functions.extractFactor(U, V, f, args.K, args.als_iter, \
				args.als_tole, args.seed)
			print(f"\rExtracted factor matrices ({round(time()-ts,1)} seconds).")
			del f, U, V

	# Estimate initial log-likelihood
	ts = time()
	l_vec = np.zeros(M)
	shared.loglike(G, P, Q, l_vec, args.threads)
	L_old = np.sum(l_vec)
	print(f"Initial loglike: {round(L_old,1)}")

	# Mini-batch parameters for stochastic EM
	guard = True
	batch = True
	batch_L = L_pre = L_old

	### EM algorithm
	# Setup containers for EM algorithm
	converged = False
	P1 = np.zeros((M, args.K))
	P2 = np.zeros((M, args.K))
	Q1 = np.zeros((N, args.K))
	Q2 = np.zeros((N, args.K))
	P_old = np.zeros((M, args.K))
	Q_old = np.zeros((N, args.K))
	Q_tmp = np.zeros((N, args.K))

	# Accelerated priming iteration
	ts = time()
	functions.steps(G, P, Q, Q_tmp, y, args.threads)
	functions.quasi(G, P, Q, Q_tmp, P1, P2, Q1, Q2, y, args.threads)
	functions.steps(G, P, Q, Q_tmp, y, args.threads)
	print(f"Performed priming iteration\t({round(time()-ts,1)}s)\n", flush=True)

	# fastmixture algorithm
	ts = time()
	print("Estimating Q and P using mini-batch EM.")
	print(f"Using {args.batches} mini-batches.")
	np.random.seed(args.seed) # Set random seed
	for it in np.arange(args.iter):
		if batch: # Quasi-Newton mini-batch updates
			B_list = np.array_split(np.random.permutation(M), args.batches)
			for b in B_list:
				functions.quasiBatch(G, P, Q, Q_tmp, P1, P2, Q1, Q2, y, np.sort(b), \
					args.threads)

			# Full updates
			if args.safety: # Safety updates
				functions.safetySteps(G, P, Q, Q_tmp, y, args.threads)
				functions.safety(G, P, Q, Q_tmp, P1, P2, Q1, Q2, y, args.threads)
				functions.safetySteps(G, P, Q, Q_tmp, y, args.threads)
			else:
				functions.quasi(G, P, Q, Q_tmp, P1, P2, Q1, Q2, y, args.threads)
		else: # Updates with log-likelihood check
			if guard:
				if args.safety:
					functions.safety(G, P, Q, Q_tmp, P1, P2, Q1, Q2, y, args.threads)
					functions.safetySteps(G, P, Q, Q_tmp, y, args.threads)
				else:
					functions.quasi(G, P, Q, Q_tmp, P1, P2, Q1, Q2, y, args.threads)
					functions.steps(G, P, Q, Q_tmp, y, args.threads)
				shared.loglike(G, P, Q, l_vec, args.threads)
				L_cur = np.sum(l_vec)
				if L_cur > L_saf:
					L_saf = L_cur
				else: # Remove guard and perform safety updates
					shared.copyP(P, P_old, args.threads)
					shared.copyQ(Q, Q_old)
					guard = False
					L_saf = L_old
			else: # Safety updates
				L_cur = functions.safetyCheck(G, P, Q, Q_tmp, P1, P2, Q1, Q2, y, \
					l_vec, L_saf, args.threads)
				if L_cur > L_saf:
					L_saf = L_cur
				else: # Break and exit with best estimate
					shared.copyP(P, P_old, args.threads)
					shared.copyQ(Q, Q_old)
					converged = True
					L_cur = L_old
					print("No improvement. Returning with best estimate!")
					print(f"Final log-likelihood: {round(L_cur,1)}")
					break
			if L_cur > L_old: # Update best guess
				shared.copyP(P_old, P, args.threads)
				shared.copyQ(Q_old, Q)
				L_old = L_cur

		# Log-likelihood convergence check
		if (it + 1) % args.check == 0:
			if batch:
				shared.loglike(G, P, Q, l_vec, args.threads)
				L_cur = np.sum(l_vec)
				L = f"({it+1})\tLog-like: {round(L_cur,1)}\t({round(time()-ts,1)}s)"
				print(L, flush=True)
				if (L_cur < L_pre) and (not args.safety): # Check unstable mini-batch
					print("Turning on safety updates.")
					shared.copyP(P, P_old, args.threads)
					shared.copyQ(Q, Q_old)
					L_cur = L_old
					batch_L = float('-inf')
					args.safety = True
				else:
					if (L_cur < batch_L) or (abs(L_cur - batch_L) < args.tole):				
						# Halve number of batches
						args.batches = args.batches//2
						if args.batches > 1:
							print(f"Using {args.batches} mini-batches.")
							L_pre = L_cur
							batch_L = float('-inf')
							if not args.safety:
								functions.steps(G, P, Q, Q_tmp, y, args.threads)
						else: # Turn off mini-batch acceleration
							print("Running standard updates.")
							batch = False
							L_saf = L_cur
							del B_list
							if not args.safety:
								functions.steps(G, P, Q, Q_tmp, y, args.threads)
					else:
						batch_L = L_cur
						if L_cur > L_old:
							shared.copyP(P_old, P, args.threads)
							shared.copyQ(Q_old, Q)
							L_old = L_cur
			else:
				L = f"({it+1})\tLog-like: {round(L_cur,1)}\t({round(time()-ts,1)}s)"
				print(L, flush=True)
				if (abs(L_cur - L_pre) < args.tole):
					if L_cur < L_old:
						shared.copyP(P, P_old, args.threads)
						shared.copyQ(Q, Q_old)
						L_cur = L_old
					converged = True
					print("Converged!")
					print(f"Final log-likelihood: {round(L_cur,1)}")
					break
				L_pre = L_cur
			ts = time()

	# Print elapsed time for estimation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")

	### Save estimates and write output to log-file
	np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.Q", Q, fmt="%.6f")
	print(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q")
	if not args.no_freqs: # Save ancestral allele frequencies
		np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.P", P, fmt="%.6f")
		print(f"Saved P matrix as {args.out}.K{args.K}.s{args.seed}.P")
	
	# Write to log-file
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
