# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawCalloc, PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log, sqrt

##### Stochastic EM algorithm #####
# Update P
cpdef void updateP(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Qa, double[:,::1] Qb, double[::1] a, \
		const long[::1] idx, const int t):
	cdef:
		int M = idx.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int d, i, j, k, x, y
		double g, h
		double* a_thr
		double* Pa_thr
		double* Pb_thr
		double* Qa_thr
		double* Qb_thr
	with nogil, parallel(num_threads=t):
		a_thr = <double*>PyMem_RawCalloc(N, sizeof(double))
		Pa_thr = <double*>PyMem_RawCalloc(K, sizeof(double))
		Pb_thr = <double*>PyMem_RawCalloc(K, sizeof(double))
		Qa_thr = <double*>PyMem_RawCalloc(N*K, sizeof(double))
		Qb_thr = <double*>PyMem_RawCalloc(N*K, sizeof(double))
		for j in prange(M):
			d = idx[j]
			for i in range(N):
				if G[d,i] != 9:
					g = <double>G[d,i]
					h = 0.0
					for k in range(K):
						h = h + Q[i,k]*P[d,k]
					for k in range(K):
						Pa_thr[k] = Pa_thr[k] + g*Q[i,k]/h
						Pb_thr[k] = Pb_thr[k] + (2-g)*Q[i,k]/(1-h)
						Qa_thr[i*K+k] = Qa_thr[i*K+k] + g*P[d,k]/h
						Qb_thr[i*K+k] = Qb_thr[i*K+k] + (2-g)*(1-P[d,k])/(1-h)
					a_thr[i] = a_thr[i] + 1.0
			for k in range(K):
				Pa_thr[k] = Pa_thr[k]*P[d,k]
				Pb_thr[k] = Pb_thr[k]*(1-P[d,k])
				P[d,k] = Pa_thr[k]/(Pa_thr[k] + Pb_thr[k])
				P[d,k] = min(max(P[d,k], 1e-5), 1-(1e-5))
				Pa_thr[k] = 0.0
				Pb_thr[k] = 0.0
		with gil:
			for x in range(N):
				a[x] += a_thr[x]
				for y in range(K):
					Qa[x,y] += Qa_thr[x*K + y]
					Qb[x,y] += Qb_thr[x*K + y]
		PyMem_RawFree(a_thr)
		PyMem_RawFree(Pa_thr)
		PyMem_RawFree(Pb_thr)
		PyMem_RawFree(Qa_thr)
		PyMem_RawFree(Qb_thr)

# Update P in acceleration
cpdef void accelP(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Qa, double[:,::1] Qb, double[:,::1] D, \
		double[::1] a, const long[::1] idx, const int t):
	cdef:
		int M = idx.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int d, i, j, k, x, y
		double g, h, P0
		double* a_thr
		double* Pa_thr
		double* Pb_thr
		double* Qa_thr
		double* Qb_thr
	with nogil, parallel(num_threads=t):
		a_thr = <double*>PyMem_RawCalloc(N, sizeof(double))
		Pa_thr = <double*>PyMem_RawCalloc(K, sizeof(double))
		Pb_thr = <double*>PyMem_RawCalloc(K, sizeof(double))
		Qa_thr = <double*>PyMem_RawCalloc(N*K, sizeof(double))
		Qb_thr = <double*>PyMem_RawCalloc(N*K, sizeof(double))
		for j in prange(M):
			d = idx[j]
			for i in range(N):
				if G[d,i] != 9:
					g = <double>G[d,i]
					h = 0.0
					for k in range(K):
						h = h + Q[i,k]*P[d,k]
					for k in range(K):
						Pa_thr[k] = Pa_thr[k] + g*Q[i,k]/h
						Pb_thr[k] = Pb_thr[k] + (2-g)*Q[i,k]/(1-h)
						Qa_thr[i*K+k] = Qa_thr[i*K+k] + g*P[d,k]/h
						Qb_thr[i*K+k] = Qb_thr[i*K+k] + (2-g)*(1-P[d,k])/(1-h)
					a_thr[i] = a_thr[i] + 1.0
			for k in range(K):
				P0 = P[d,k]
				Pa_thr[k] = Pa_thr[k]*P[d,k]
				Pb_thr[k] = Pb_thr[k]*(1-P[d,k])
				P[d,k] = Pa_thr[k]/(Pa_thr[k] + Pb_thr[k])
				P[d,k] = min(max(P[d,k], 1e-5), 1-(1e-5))
				D[d,k] = P[d,k] - P0
				Pa_thr[k] = 0.0
				Pb_thr[k] = 0.0
		with gil:
			for x in range(N):
				a[x] += a_thr[x]
				for y in range(K):
					Qa[x,y] += Qa_thr[x*K + y]
					Qb[x,y] += Qb_thr[x*K + y]
		PyMem_RawFree(a_thr)
		PyMem_RawFree(Pa_thr)
		PyMem_RawFree(Pb_thr)
		PyMem_RawFree(Qa_thr)
		PyMem_RawFree(Qb_thr)

# Accelerated jump for P (SQUAREM)
cpdef void alphaP(double[:,::1] P, const double[:,::1] P0, const double[:,::1] D1, \
		const double[:,::1] D2, double[:,::1] D3, const long[::1] idx, const int t) \
		noexcept nogil:
	cdef:
		int M = idx.shape[0]
		int K = P.shape[1]
		int d, j, k
		double sum1 = 0.0
		double sum2 = 0.0
		double alpha
	for j in range(M):
		d = idx[j]
		for k in range(K):
			D3[d,k] = D2[d,k] - D1[d,k]
			sum1 += D1[d,k]*D1[d,k]
			sum2 += D3[d,k]*D3[d,k]
	alpha = max(1.0, sqrt(sum1)/sqrt(sum2))
	for j in prange(M, num_threads=t):
		d = idx[j]
		for k in range(K):
			P[d,k] = P0[d,k] + 2.0*alpha*D1[d,k] + alpha*alpha*D3[d,k]
			P[d,k] = min(max(P[d,k], 1e-5), 1-(1e-5))
