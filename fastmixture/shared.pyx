# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log, sqrt

##### fastmixture ######
# Expand data into full genotype matrix
cpdef void expandGeno(const unsigned char[::1] B, unsigned char[:,::1] G, \
		const int N_bytes, const int t) noexcept nogil:
	cdef:
		int M = G.shape[0]
		int N = G.shape[1]
		int i, j, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(N_bytes):
			byte = B[j*N_bytes + b]
			for bytepart in range(4):
				G[j,i] = recode[byte & mask]
				byte = byte >> 2
				i = i + 1
				if i == N:
					break

# Estimate minor allele frequencies
cpdef void estimateFreq(const unsigned char[:,::1] G, double[::1] f, \
		const int t) noexcept nogil:
	cdef:
		int M = G.shape[0]
		int N = G.shape[1]
		int i, j
		double n
	for j in prange(M, num_threads=t):
		n = 0.0
		for i in range(N):
			f[j] += <double>G[j,i]
			n = n + 1.0
		f[j] /= (2.0*n)

# Log-likelihood
cpdef void loglike(const unsigned char[:,::1] G, const double[:,::1] P, \
		const double[:,::1] Q, double[::1] l_vec, const int t) noexcept nogil:
	cdef:
		int M = G.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int i, j, k
		double g, h
	for j in prange(M, num_threads=t):
		l_vec[j] = 0.0
		for i in range(N):
			g = <double>G[j,i]
			h = 0.0
			for k in range(K):
				h = h + Q[i,k]*P[j,k]
			l_vec[j] += g*log(h) + (2.0-g)*log(1.0-h)

# Root-mean-square error
cpdef double rmse(const double[:,::1] Q, const double[:,::1] Q_pre) noexcept nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double r = 0.0
	for i in range(N):
		for k in range(K):
			r += (Q[i,k] - Q_pre[i,k])*(Q[i,k] - Q_pre[i,k])
	return sqrt(r/<double>(N*K))

# Sum-of-squares used for evaluation 
cpdef void sumSquare(const unsigned char[:,::1] G, const double[:,::1] P, \
		const double[:,::1] Q, double[::1] l_vec, const int t) noexcept nogil:
	cdef:
		int M = G.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int i, j, k
		double h, g
	for j in prange(M, num_threads=t):
		l_vec[j] = 0.0
		for i in range(N):
			g = <double>G[j,i]
			h = 0.0
			for k in range(K):
				h = h + Q[i,k]*P[j,k]
			l_vec[j] += (g-2.0*h)*(g-2.0*h)
