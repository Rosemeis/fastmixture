# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### Randomized SVD - PCAone method #####
# Load centered chunk of PLINK file for SVD (individual allele frequencies)
cpdef void plinkChunk(unsigned char[:,::1] G, double[:,::1] X, double[::1] f, \
		int M_b, int t):
	cdef:
		int M = X.shape[0]
		int N = X.shape[1]
		int B = G.shape[1]
		int i, j, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	with nogil:
		for j in prange(M, num_threads=t):
			i = 0
			for b in range(B):
				byte = G[M_b+j,b]
				for bytepart in range(4):
					if recode[byte & mask] != 9:
						X[j,i] = <double>recode[byte & mask] - 2.0*f[M_b+j]
					else:
						X[j,i] = 0.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break

# Root-mean square error between two Q matrices
cpdef double rmse(double[:,::1] A, double[:,::1] B):
	cdef:
		int N = A.shape[0]
		int K = A.shape[1]
		int i, k
		double res = 0.0
	for i in range(N):
		for k in range(K):
			res += (A[i,k] - B[i,k])*(A[i,k] - B[i,k])
	return sqrt(res/<double>(N*K))
