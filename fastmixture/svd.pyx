# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from libc.math cimport fmaxf, fminf, sqrtf
from libc.stdint cimport uint8_t

cdef float PRO_MIN = 1e-5
cdef float PRO_MAX = 1.0-(1e-5)

##### Randomized SVD - PCAone method #####
# Truncate parameters to domain
cdef inline float _project(
		float s
	) noexcept nogil:
	return fminf(fmaxf(s, PRO_MIN), PRO_MAX)

# Load centered chunk from PLINK file for SVD
cpdef void plinkChunk(
		const uint8_t[:,::1] G, float[:,::1] X, const float[::1] f, const size_t M_b
	) noexcept nogil:
	cdef:
		size_t M = X.shape[0]
		size_t N = X.shape[1]
		size_t i, j, l
		float u
	for j in prange(M):
		l = M_b+j
		u = 2.0*f[l]
		for i in range(N):
			if G[l,i] != 9:
				X[j,i] = <float>G[l,i] - u
			else:
				X[j,i] = 0.0

# Root-mean square error between two Q matrices
cpdef float rmse(
		const float[:,::1] A, const float[:,::1] B
	) noexcept nogil:
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
cpdef void map2domain(
		float[:,::1] Q
	) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		float sumQ
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = _project(Q[i,k])
			sumQ += Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ
