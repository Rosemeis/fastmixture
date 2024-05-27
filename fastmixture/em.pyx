# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawCalloc, PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log, sqrt

##### fastmixture #####
# Update P and temp Q arrays
cpdef void updateP(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_new, const int t) noexcept nogil:
	cdef:
		int M = G.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int i, j, k, x, y
		double a, b, g, h
		double* P_thr
		double* Q_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>PyMem_RawCalloc(2*K, sizeof(double))
		Q_thr = <double*>PyMem_RawCalloc(N*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				g = <double>G[j,i]
				h = 0.0
				for k in range(K):
					h = h + Q[i,k]*P[j,k]
				a = g/h
				b = (2.0-g)/(1.0-h)
				for k in range(K):
					P_thr[k] = P_thr[k] + Q[i,k]*a
					P_thr[K+k] = P_thr[K+k] + Q[i,k]*b
					Q_thr[i*K+k] = Q_thr[i*K+k] + P[j,k]*a + (1.0-P[j,k])*b
			for k in range(K):
				P_thr[k] = P_thr[k]*P[j,k]
				P[j,k] = P_thr[k]/(P_thr[k] + P_thr[K+k]*(1-P[j,k]))
				P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))
				P_thr[k] = 0.0
				P_thr[K+k] = 0.0
		with gil:
			for x in range(N):
				for y in range(K):
					Q_new[x,y] += Q_thr[x*K + y]
		PyMem_RawFree(P_thr)
		PyMem_RawFree(Q_thr)

# Update P in acceleration
cpdef void accelP(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] D, \
		const int t) noexcept nogil:
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = P.shape[1]
		int i, j, k, x, y
		double a, b, g, h, p
		double* P_thr
		double* Q_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>PyMem_RawCalloc(2*K, sizeof(double))
		Q_thr = <double*>PyMem_RawCalloc(N*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				g = <double>G[j,i]
				h = 0.0
				for k in range(K):
					h = h + Q[i,k]*P[j,k]
				a = g/h
				b = (2.0-g)/(1.0-h)
				for k in range(K):
					P_thr[k] = P_thr[k] + Q[i,k]*a
					P_thr[K+k] = P_thr[K+k] + Q[i,k]*b
					Q_thr[i*K+k] = Q_thr[i*K+k] + P[j,k]*a + (1.0-P[j,k])*b
			for k in range(K):
				p = P[j,k]
				P_thr[k] = P_thr[k]*P[j,k]
				P[j,k] = P_thr[k]/(P_thr[k] + P_thr[K+k]*(1-P[j,k]))
				P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))
				D[j,k] = P[j,k] - p
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
		const double[:,::1] D2, double[:,::1] D3, const int t) noexcept nogil:
	cdef:
		int M = P.shape[0]
		int K = P.shape[1]
		int j, k
		double sum1 = 0.0
		double sum2 = 0.0
		double alpha, a1, a2
	for j in range(M):
		for k in range(K):
			D3[j,k] = D2[j,k] - D1[j,k]
			sum1 += D1[j,k]*D1[j,k]
			sum2 += D3[j,k]*D3[j,k]
	alpha = max(1.0, sqrt(sum1)/sqrt(sum2))
	a1 = alpha*2.0
	a2 = alpha*alpha
	for j in prange(M, num_threads=t):
		for k in range(K):
			P[j,k] = P0[j,k] + a1*D1[j,k] + a2*D3[j,k]
			P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))

# Update Q
cpdef void updateQ(double[:,::1] Q, double[:,::1] Q_new, const int M, const int t) \
		noexcept nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k
		double a = 1.0/<double>(2*M)
		double sumQ
	for i in prange(N, num_threads=t):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] *= Q_new[i,k]*a
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			Q_new[i,k] = 0.0
			sumQ = sumQ + Q[i,k]
		# map2domain (normalize)
		for k in range(K):
			Q[i,k] /= sumQ

# Update Q in acceleration
cpdef void accelQ(double[:,::1] Q, double[:,::1] Q_new, double[:,::1] D, \
		const int M, const int t) noexcept nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k
		double a = 1.0/<double>(2*M)
		double sumQ
		double* q_thr
	with nogil, parallel(num_threads=t):
		q_thr = <double*>PyMem_RawMalloc(sizeof(double)*K)
		for i in prange(N):
			sumQ = 0.0
			for k in range(K):
				q_thr[k] = Q[i,k]
				Q[i,k] *= Q_new[i,k]*a
				Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
				Q_new[i,k] = 0.0
				sumQ = sumQ + Q[i,k]
			# map2domain (normalize)
			for k in range(K):
				Q[i,k] /= sumQ
				D[i,k] = Q[i,k] - q_thr[k]
		PyMem_RawFree(q_thr)

# Accelerated jump for Q (SQUAREM)
cpdef void alphaQ(double[:,::1] Q, const double[:,::1] Q0, const double[:,::1] D1, \
		const double[:,::1] D2, double[:,::1] D3, const int t) noexcept nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sum1 = 0.0
		double sum2 = 0.0
		double sumQ
		double alpha, a1, a2
	for i in range(N):
		for k in range(K):
			D3[i,k] = D2[i,k] - D1[i,k]
			sum1 += D1[i,k]*D1[i,k]
			sum2 += D3[i,k]*D3[i,k]
	alpha = max(1.0, sqrt(sum1)/sqrt(sum2))
	a1 = alpha*2.0
	a2 = alpha*alpha
	for i in prange(N, num_threads=t):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = Q0[i,k] + a1*D1[i,k] + a2*D3[i,k]
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			sumQ = sumQ + Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ
