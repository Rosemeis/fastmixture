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
parser.add_argument("--inverse", action="store_true",
	help="Inverse the P file due to genotype coding")
parser.add_argument("--loglike", action="store_true",
	help="Log-likelihood estimates")
parser.add_argument("--sumsquares", action="store_true",
	help="Sum-of-squares estimates")
parser.add_argument("--rmse", action="store_true",
	help="Root mean-square-error to ground truth")
parser.add_argument("--jsd", action="store_true",
	help="Jensen-Shannon divergence to ground-truth")
parser.add_argument("--tfile",
	help="Path to ground truth Q-file")
parser.add_argument("--labels",
	help="Path to labels for population-specific measures")

# Check input
args = parser.parse_args()
assert args.qfile is not None, "No ancestry proportions (--qfile)!"
if args.rmse or args.jsd:
	assert args.tfile is not None, "No ground truth (--tfile)!"
else:
	assert args.bfile is not None, "No input data (--bfile)!"
	assert args.pfile is not None, "No frequencies (--pfile)!"
	assert (args.loglike or args.sumsquares), "No valid option chosen (--loglike, --sumsquares)!"

# Control threads of external numerical libraries
os.environ["MKL_NUM_THREADS"] = str(args.threads)
os.environ["MKL_MAX_THREADS"] = str(args.threads)
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["OMP_MAX_THREADS"] = str(args.threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

# Import numerical libraries
import numpy as np
from fastmixture import shared
from fastmixture import utils

### Read data
# Read Q file
Q = np.loadtxt(f"{args.qfile}", dtype=float)
if Q.shape[1] > Q.shape[0]:
	Q = np.ascontiguousarray(Q.T)
Q.clip(min=args.bound, max=1-(args.bound), out=Q)
Q /= np.sum(Q, axis=1, keepdims=True)
K = Q.shape[1]

# Load data files needed
if args.rmse or args.jsd:
	# Ground truth Q file
	S = np.loadtxt(f"{args.tfile}", dtype=float)
else:
	# Read PLINK files
	G, _, M, N = utils.readPlink(args.bfile)

	# Read P file
	P = np.loadtxt(f"{args.pfile}", dtype=float)
	if args.inverse:
		P = 1 - P
	assert P.shape[0] == M, "Number of SNPs doesn't match!"
	assert P.shape[1] == K, "Number of components doesn't match!"
	P.clip(min=args.bound, max=1-(args.bound), out=P)

### Evaluation
if args.rmse or args.jsd:
	# Find best matching pairs between the two files
	q_list = []
	s_list = []
	d_mat = np.zeros((K, K))
	for k1 in range(K):
		for k2 in range(K):
			d_mat[k1,k2] = np.dot(Q[:,k1]-S[:,k2], Q[:,k1]-S[:,k2])
	
	# Loop over indices
	for k in range(K*K):
		i1, i2 = np.unravel_index(np.argsort(d_mat.flatten())[k], (K,K))
		if (i1 not in q_list) and (i2 not in s_list):
			q_list.append(i1)
			s_list.append(i2)
		if len(q_list) == K:
			break

	# Reorder and compute metric
	Q = np.ascontiguousarray(Q[:,q_list])
	S = np.ascontiguousarray(S[:,s_list])
	if args.labels is None: # Compute metric on full dataset
		if args.rmse:
			print(f"{shared.rmse(Q, S):.7f}")
		else:
			jsd = (shared.divKL(Q, S) + shared.divKL(S, Q))*0.5
			print(f"{jsd:.7f}")
	else:  # Compute metric in specific populations/partitions
		Y = np.loadtxt(f"{args.labels}", dtype=np.uint8)
		U = np.unique(Y)
		for i, y in enumerate(U):
			Q_sub = Q[Y == y,:]
			S_sub = S[Y == y,:]
			if args.rmse:
				print(f"Label {i}:\t{shared.rmse(Q_sub, S_sub):.7f}")
			else:
				jsd = (shared.divKL(Q_sub, S_sub) + shared.divKL(S_sub, Q_sub))*0.5
				print(f"Label {i}:\t{jsd:.7f}")
else:
	if args.loglike: # Log-likelihood
		L = shared.loglike(G, P, Q)
	else: # Sum-of-squares
		L = shared.sumSquare(G, P, Q)
	print(f"{round(L,1)}", flush=True)
