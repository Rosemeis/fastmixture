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

VERSION = "1.0.0"

### Argparse
parser = argparse.ArgumentParser(prog="fastmixture")
parser.add_argument("--version", action="version",
	version=f"v{VERSION}")
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
parser.add_argument("--projection", metavar="FILE",
	help="Path to ancestral allele frequencies file")
parser.add_argument("--check", metavar="INT", type=int, default=5,
	help="Number of iterations between check for convergence")
parser.add_argument("--power", metavar="INT", type=int, default=10,
	help="Number of power iterations in randomized SVD (10)")
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



##### fastmixture #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print("-------------------------------------------------")
	print(f"fastmixture v{VERSION}")
	print("C.G. Santander, A. Refoyo-Martinez and J. Meisner")
	print(f"K={args.K}, seed={args.seed}, batches={args.batches}, threads={args.threads}")
	print("-------------------------------------------------\n")

	# Check input
	assert args.bfile is not None, "No input data (--bfile)!"
	assert args.K > 1, "Please select K > 1!"
	assert args.threads > 0, "Please select a valid number of threads!"
	assert args.seed >= 0, "Please select a valid seed!"
	assert args.batches > 1, "Please select a valid number of batches!"
	assert args.iter > 0, "Please select a valid number of iterations!"
	assert args.tole >= 0.0, "Please select a valid tolerance!"
	assert args.check > 0, "Please select a valid value for convergence check!"
	assert args.power > 0, "Please select a valid number of power iterations!"
	assert args.chunk > 0, "Please select a valid value for chunk size in SVD!"
	assert args.als_iter > 0, "Please select a valid number of iterations in ALS!"
	assert args.als_tole >= 0.0, "Please select a valid tolerance in ALS!"
	start = time()

	# Create log-file of used arguments
	full = vars(parser.parse_args())
	deaf = vars(parser.parse_args([]))
	mand = ["seed", "batches"]
	with open(f"{args.out}.K{args.K}.s{args.seed}.log", "w") as log:
		log.write(f"fastmixture v{VERSION}\n")
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
	os.environ["MKL_MAX_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_MAX_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

	# Load numerical libraries
	import numpy as np
	from fastmixture import utils
	from fastmixture import shared

	### Read data
	assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
	assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
	print("Reading data...", end="", flush=True)
	G, q_nrm, M, N = utils.readPlink(args.bfile)
	assert not np.any(q_nrm == 0), "Sample(s) with zero information!"
	rng = np.random.default_rng(args.seed) # Set up random number generator
	print(f"\rLoaded {N} samples and {M} SNPs.")

	# Set up parameters
	if args.supervised is not None: # Supervised mode
		# Check input of ancestral sources
		print("Ancestry estimation in supervised mode!")
		y = np.loadtxt(args.supervised, dtype=np.uint8).reshape(-1)
		assert y.shape[0] == N, "Number of samples differ between files!"
		assert np.max(y) <= args.K, "Wrong number of ancestral sources!"
		assert np.min(y) >= 0, "Wrong format in population assignments!"
		print(f"{np.sum(y > 0)}/{N} individuals with fixed ancestry.")

		# Count ancestral sources
		z = np.unique(y[y > 0])
		z -= 1
		z = np.sort(z)

		# Initialize parameters
		P = rng.random(size=(M, args.K))
		Q = rng.random(size=(N, args.K))
		P[:,z] = 0.0
		shared.initP(G, P, y)
		shared.initQ(Q, y)
		del z
	elif args.projection is not None: # Projection mode
		# Check input of ancestral allele frequencies
		print("Ancestry estimation in projection mode!")
		P = np.loadtxt(args.projection, dtype=float).clip(min=1e-5, max=1.0-(1e-5))
		assert P.shape[0] == M, "Number of SNPs differ between files!"
		assert P.shape[1] == args.K, "Wrong number of ancestral sources!"

		# Initialize Q matrix
		Q = rng.random(size=(N, args.K)).clip(min=1e-5, max=1.0-(1e-5))
		Q /= np.sum(Q, axis=1, keepdims=True)
	else: # Standard unsupervised mode
		if args.random_init: # Random initialization
			print("Random initialization.")
			P = rng.random(size=(M, args.K)).clip(min=1e-5, max=1.0-(1e-5))
			Q = rng.random(size=(N, args.K)).clip(min=1e-5, max=1.0-(1e-5))
			Q /= np.sum(Q, axis=1, keepdims=True)
		else: # SVD-based initialization
			f = np.zeros(M, dtype=np.float32)
			shared.estimateFreq(G, f)
			assert (np.min(f) > 0.0) & (np.max(f) < 1.0), "Please perform MAF filtering!"

			# Initialize P and Q matrices from SVD and ALS
			ts = time()
			print("Performing SVD and ALS.", end="", flush=True)
			U, V = utils.randomizedSVD(G, f, args.K-1, args.chunk, args.power, rng)
			P, Q = utils.extractFactor(U, V, f, args.K, args.als_iter, \
				args.als_tole, rng)
			print(f"\rExtracted factor matrices\t({time()-ts:.1f}s)")
			del f, U, V
		y = None

	# Set up containers for EM algorithm
	s = np.arange(M, dtype=np.uint32)
	Q1 = np.zeros_like(Q)
	Q2 = np.zeros_like(Q)
	Q_old = np.zeros_like(Q)
	Q_tmp = np.zeros_like(Q)
	q_bat = np.zeros(N)
	if args.projection is None:
		P1 = np.zeros_like(P)
		P2 = np.zeros_like(P)
		P_old = np.zeros_like(P)

	# Fastmixture
	if args.projection is not None: # Projection mode
		from fastmixture import projection
		L, it, con = projection.fastProject(G, P, Q, Q1, Q2, Q_tmp, Q_old, q_nrm, \
			q_bat, s, args.iter, args.tole, args.check, args.batches, rng)
	else: # Unsupervised or supervised mode
		from fastmixture import functions
		L, it, con = functions.fastRun(G, P, Q, P1, P2, Q1, Q2, Q_tmp, P_old, Q_old, \
			q_nrm, q_bat, s, y, args.iter, args.tole, args.check, args.batches, rng)

	# Print elapsed time for estimation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")

	### Save estimates and write output to log-file
	np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.Q", Q, fmt="%.6f")
	print(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q")
	if not args.no_freqs or (args.projection is not None): # Save frequencies
		np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.P", P, fmt="%.6f")
		print(f"Saved P matrix as {args.out}.K{args.K}.s{args.seed}.P")

	# Write to log-file
	with open(f"{args.out}.K{args.K}.s{args.seed}.log", "a") as log:
		log.write(f"\nFinal log-likelihood: {L:.1f}\n")
		if con:
			log.write(f"Converged in {it+1} iterations.\n")
		else:
			log.write(f"EM algorithm did not converge in {args.iter} iterations!\n")
		log.write(f"Total elapsed time: {t_min}m{t_sec}s\n")
		log.write(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q\n")
		if not args.no_freqs or (args.projection is not None):
			log.write(f"Saved P matrix as {args.out}.K{args.K}.s{args.seed}.P\n")



##### Main exception #####
assert __name__ != "__main__", "Please use the 'fastmixture' command!"
