# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawCalloc, PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log, sqrt

##### Stochastic EM algorithm #####
# Update P in acceleration
cpdef void accelP(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] D, \
		const long[::1] s, const int t) noexcept nogil:
	cdef:
		int M = s.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int d, i, j, k, x, y
		double a, b, g, h, p
		double* P_thr
		double* Q_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>PyMem_RawCalloc(2*K, sizeof(double))
		Q_thr = <double*>PyMem_RawCalloc(N*K, sizeof(double))
		for j in prange(M):
			d = s[j]
			for i in range(N):
				g = <double>G[d,i]
				h = 0.0
				for k in range(K):
					h = h + Q[i,k]*P[d,k]
				a = g/h
				b = (2.0-g)/(1.0-h)
				for k in range(K):
					P_thr[k] = P_thr[k] + Q[i,k]*a
					P_thr[K+k] = P_thr[K+k] + Q[i,k]*b
					Q_thr[i*K+k] = Q_thr[i*K+k] + P[d,k]*a + (1.0-P[d,k])*b
			for k in range(K):
				p = P[d,k]
				P_thr[k] = P_thr[k]*P[d,k]
				P[d,k] = P_thr[k]/(P_thr[k] + P_thr[K+k]*(1-P[d,k]))
				P[d,k] = min(max(P[d,k], 1e-5), 1-(1e-5))
				D[d,k] = P[d,k] - p
				P_thr[k] = 0.0
				P_thr[K+k] = 0.0
		with gil:
			for x in range(N):
				for y in range(K):
					Q_new[x,y] += Q_thr[x*K + y]
		PyMem_RawFree(P_thr)
		PyMem_RawFree(Q_thr)

# Accelerated jump for P (SQUAREM)
cpdef void alphaP(double[:,::1] P, const double[:,::1] P0, const double[:,::1] D1, \
		const double[:,::1] D2, double[:,::1] D3, const long[::1] s, const int t) \
		noexcept nogil:
	cdef:
		int M = s.shape[0]
		int K = P.shape[1]
		int d, j, k
		double sum1 = 0.0
		double sum2 = 0.0
		double alpha
	for j in range(M):
		d = s[j]
		for k in range(K):
			D3[d,k] = D2[d,k] - D1[d,k]
			sum1 += D1[d,k]*D1[d,k]
			sum2 += D3[d,k]*D3[d,k]
	alpha = max(1.0, sqrt(sum1)/sqrt(sum2))
	for j in prange(M, num_threads=t):
		d = s[j]
		for k in range(K):
			P[d,k] = P0[d,k] + 2.0*alpha*D1[d,k] + alpha*alpha*D3[d,k]
			P[d,k] = min(max(P[d,k], 1e-5), 1-(1e-5))
