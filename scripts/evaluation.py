"""
Generate evaluation estimates for ancestry estimation.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os

### Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bfile",
	help="Prefix for PLINK files (.bed, .bim, .fam)")
parser.add_argument("-q", "--qfile",
	help="Path to Q-file")
parser.add_argument("-p", "--pfile",
	help="Path to P-file")
parser.add_argument("-t", "--threads", type=int, default=1,
	help="Number of threads (1)")
parser.add_argument("--bound", type=float, default=1e-5,
	help="Edge bound for 0 and 1 (1e-5)")
parser.add_argument("--scope", action="store_true",
	help="Indicator for SCOPE output files")
parser.add_argument("--loglike", action="store_true",
	help="Log-likelihood estimates")
parser.add_argument("--sumsquares", action="store_true",
	help="Sum-of-squares estimates")
parser.add_argument("--rmse", action="store_true",
	help="Root mean-square-error to ground truth")
parser.add_argument("--tfile",
	help="Path to ground truth Q-file")

# Check input
args = parser.parse_args()
assert args.qfile is not None, "No ancestry proportions (--qfile)!"
if args.rmse:
	assert args.tfile is not None, "No ground truth (--tfile)!"
else:
	assert args.bfile is not None, "No input data (--bfile)!"
	assert args.pfile is not None, "No frequencies (--pfile)!"
	assert (args.loglike or args.sumsquares), "No valid option chosen " + \
		"(--loglike, --sumsquares)!"

# Control threads of external numerical libraries
os.environ["MKL_NUM_THREADS"] = str(args.threads)
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

# Import numerical libraries
import numpy as np
from math import ceil
from fastmixture import shared
from fastmixture import functions

### Read data
# Read Q file
Q = np.loadtxt(f"{args.qfile}", dtype=float)
if args.scope:
	Q = np.ascontiguousarray(Q.T)
Q.clip(min=args.bound, max=1-(args.bound), out=Q)
Q /= np.sum(Q, axis=1, keepdims=True)

# Load data files needed
if args.rmse:
	# Ground truth Q file
	S = np.loadtxt(f"{args.tfile}", dtype=float)
else:
	# Finding length of .fam and .bim file and finding chromosome indices
	N = functions.extract_length(f"{args.bfile}.fam")
	M = functions.extract_length(f"{args.bfile}.bim")
	G = np.zeros((M, N), dtype=np.uint8)
	assert Q.shape[0] == N, "Number of individuals doesn't match!"

	# Read .bed file
	with open(f"{args.bfile}.bed", "rb") as bed:
		B = np.fromfile(bed, dtype=np.uint8, offset=3)
	N_bytes = ceil(N/4) # Length of bytes to describe N individuals
	shared.expandGeno(B, G, N_bytes, args.threads)
	del B

	# Read P file
	P = np.loadtxt(f"{args.pfile}", dtype=float)
	if args.scope:
		P = 1 - P
	assert P.shape[0] == M, "Number of SNPs doesn't match!"
	P.clip(min=args.bound, max=1-(args.bound), out=P)

	# Initalize parameters
	l_vec = np.zeros(M)

### Evaluation
if args.rmse:
	c = np.arange(Q.shape[1], dtype=int)
	for k1 in range(Q.shape[1]):
		v = np.inf
		for k2 in range(Q.shape[1]):
			d = np.sqrt(np.dot(Q[:,k1] - S[:,k2], Q[:,k1] - S[:,k2])/float(N))
			if d < v:
				v = d
				c[k1] = k2
	S_new = np.ascontiguousarray(S[:,c])
	print(f"{shared.rmse(Q, S_new):.7f}")
else:
	if args.loglike: # Log-likelihood
		shared.loglike(G, P, Q, l_vec, args.threads)
	else: # Sum-of-squares
		shared.sumSquare(G, P, Q, l_vec, args.threads)
	L = np.sum(l_vec)
	print(f"{round(L,1)}", flush=True)
