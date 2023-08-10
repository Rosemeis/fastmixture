"""
Generate sum-of-squares estimates for ancestry estimation.
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
	help="Number of threads")
parser.add_argument("--bound", type=float, default=1e-5,
	help="Edge bound for 0 and 1")
parser.add_argument("--scope", action="store_true",
	help="Indicator for SCOPE output files")

# Check input
args = parser.parse_args()
assert args.bfile is not None, "No input data (--bfile)!"
assert args.qfile is not None, "No ancestry proportions (--qfile)!"
assert args.pfile is not None, "No ancestral frequencies (--pfile)!"

# Control threads of external numerical libraries
os.environ["MKL_NUM_THREADS"] = str(args.threads)
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

# Import numerical libraries
import numpy as np
from math import ceil
from src import svd
from src import functions

### Read data
# Finding length of .fam and .bim file and finding chromosome indices
N = functions.extract_length(f"{args.bfile}.fam")
M = functions.extract_length(f"{args.bfile}.bim")

# Read .bed file
with open(f"{args.bfile}.bed", "rb") as bed:
	G = np.fromfile(bed, dtype=np.uint8, offset=3)
B = ceil(N/4) # Length of bytes to describe n individuals
G.shape = (M, B)

### Initalize parameters
f = np.zeros(M)
lsVec = np.zeros(M)

# Load Q and P file
Q = np.loadtxt(f"{args.qfile}", dtype=float)
if args.scope:
	Q = np.ascontiguousarray(Q.T)
P = np.loadtxt(f"{args.pfile}", dtype=float)
if args.scope:
	P = 1 - P
assert Q.shape[0] == N, "Number of individuals doesn't match!"
assert P.shape[0] == M, "Number of SNPs doesn't match!"

# Map to bound
Q.clip(min=args.bound, max=1-(args.bound), out=Q)
Q /= np.sum(Q, axis=1, keepdims=True)
P.clip(min=args.bound, max=1-(args.bound), out=P)

### Estimate least square estimate
svd.sumSquare(G, P, Q, lsVec, args.threads)
ls = np.sum(lsVec)
print(f"{round(ls,1)}", flush=True)
