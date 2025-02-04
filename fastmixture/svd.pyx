# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from libc.math cimport fmaxf, fminf, sqrtf

##### Randomized SVD - PCAone method #####
# Truncate parameters to domain
cdef inline float project(float s) noexcept nogil:
	cdef:
		float min_val = 1e-5
		float max_val = 1.0-(1e-5)
	return fminf(fmaxf(s, min_val), max_val)

# Load centered chunk from PLINK file for SVD
cpdef void plinkChunk(const unsigned char[:,::1] G, float[:,::1] X, \
		const float[::1] f, const size_t M_b) noexcept nogil:
	cdef:
		size_t M = X.shape[0]
		size_t N = X.shape[1]
		size_t i, j, l
		unsigned char g
		float fl
	for j in prange(M):
		l = M_b+j
		fl = f[l]
		for i in range(N):
			g = G[l,i]
			if g != 9:
				X[j,i] = <float>g - 2.0*fl
			else:
				X[j,i] = 0.0

# Root-mean square error between two Q matrices
cpdef float rmse(const float[:,::1] A, const float[:,::1] B) noexcept nogil:
	cdef:
		size_t N = A.shape[0]
		size_t K = A.shape[1]
		size_t i, k
		float s = 1.0/<float>(N*K)
		float res = 0.0
	for i in range(N):
		for k in range(K):
			res += (A[i,k] - B[i,k])*(A[i,k] - B[i,k])
	return sqrtf(res*s)

# Map Q parameters to domain
cpdef void map2domain(float[:,::1] Q) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		float sumQ, valQ
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			valQ = project(Q[i,k])
			sumQ += valQ
			Q[i,k] = valQ
		for k in range(K):
			Q[i,k] /= sumQ
