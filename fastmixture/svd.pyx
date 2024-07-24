# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### Randomized SVD - PCAone method #####
# Inline function
cdef inline double project(double s) noexcept nogil:
	return min(max(s, 1e-5), 1-(1e-5))

# Load centered chunk from PLINK file for SVD
cpdef void plinkChunk(const unsigned char[:,::1] G, double[:,::1] X, \
		const double[::1] f, const int M_b, const int t) noexcept nogil:
	cdef:
		int M = X.shape[0]
		int N = X.shape[1]
		int i, j
	for j in prange(M, num_threads=t):
		for i in range(N):
			X[j,i] = <double>G[M_b+j,i] - 2.0*f[M_b+j]

# Root-mean square error between two Q matrices
cpdef double rmse(const double[:,::1] A, const double[:,::1] B) noexcept nogil:
	cdef:
		int N = A.shape[0]
		int K = A.shape[1]
		int i, k
		double s = 1.0/<double>(N*K)
		double res = 0.0
	for i in range(N):
		for k in range(K):
			res += (A[i,k] - B[i,k])*(A[i,k] - B[i,k])
	return sqrt(res*s)

# Map2domain
cpdef void map2domain(double[:,::1] Q) noexcept nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sumQ
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = project(Q[i,k])
			sumQ += Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ
