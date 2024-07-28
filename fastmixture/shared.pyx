# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log, sqrt

##### fastmixture ######
# Inline function
cdef inline double computeH(const double* p, const double* q, int K) noexcept nogil:
	cdef:
		int k
		double h = 0.0
	for k in range(K):
		h += p[k]*q[k]
	return h

# Expand data into full genotype matrix
cpdef void expandGeno(const unsigned char[:,::1] B, unsigned char[:,::1] G, \
		const int t) noexcept nogil:
	cdef:
		int M = G.shape[0]
		int N = G.shape[1]
		int N_b = B.shape[1]
		int i, j, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		for b in range(N_b):
			byte = B[j,b]
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
			h = computeH(&P[j,0], &Q[i,0], K)
			l_vec[j] += g*log(h) + (2.0-g)*log(1.0-h)

# Copy P array
cpdef void copyP(double[:,::1] P0, const double[:,::1] P1, const int t) noexcept nogil:
	cdef:
		int M = P0.shape[0]
		int K = P0.shape[1]
		int j, k
	for j in prange(M, num_threads=t):
		for k in range(K):
			P0[j,k] = P1[j,k]

# Copy Q array
cpdef void copyQ(double[:,::1] Q0, const double[:,::1] Q1) noexcept nogil:
	cdef:
		int N = Q0.shape[0]
		int K = Q0.shape[1]
		int i, k
	for i in range(N):
		for k in range(K):
			Q0[i,k] = Q1[i,k]

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

# Sum-of-squares used in evaluation 
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
			h = computeH(&P[j,0], &Q[i,0], K)
			l_vec[j] += (g-2.0*h)*(g-2.0*h)

# Kullback-Leibler divergence with average for Jensen-Shannon
cpdef double divKL(const double[:,::1] A, const double[:,::1] B) noexcept nogil:
	cdef:
		int N = A.shape[0]
		int K = A.shape[1]
		int i, k
		double d = 0.0
		double a
	for i in range(N):
		for k in range(K):
			a = (A[i,k] + B[i,k])*0.5
			d += A[i,k]*log(A[i,k]/a + 1e-9)
	return d/<double>N
