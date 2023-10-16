# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log, sqrt

##### fastmixture #####
# Update P and temp Q arrays
cpdef void updateP(unsigned char[:,::1] G, double[:,::1] P, double[:,::1] Q, \
		double[:,::1] Qa, double[:,::1] Qb, double[::1] a, int t):
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k, x, y, i0, k0, b, bytepart
		double g, h
		double* a_thr
		double* Pa_thr
		double* Pb_thr
		double* Qa_thr
		double* Qb_thr
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	with nogil, parallel(num_threads=t):
		a_thr = <double*>PyMem_RawMalloc(sizeof(double)*N)
		Pa_thr = <double*>PyMem_RawMalloc(sizeof(double)*K)
		Pb_thr = <double*>PyMem_RawMalloc(sizeof(double)*K)
		Qa_thr = <double*>PyMem_RawMalloc(sizeof(double)*N*K)
		Qb_thr = <double*>PyMem_RawMalloc(sizeof(double)*N*K)
		for i0 in range(N):
			a_thr[i0] = 0.0
			for k0 in range(K):
				Qa_thr[i0*K + k0] = 0.0
				Qb_thr[i0*K + k0] = 0.0
		for j in prange(M):
			for k in range(K):
				Pa_thr[k] = 0.0
				Pb_thr[k] = 0.0
			i = 0
			for b in range(B):
				byte = G[j,b]
				for bytepart in range(4):
					if recode[byte & mask] != 9:
						g = <double>recode[byte & mask]
						h = 0.0
						for k in range(K):
							h = h + Q[i,k]*P[j,k]
						for k in range(K):
							Pa_thr[k] = Pa_thr[k] + g*Q[i,k]/h
							Pb_thr[k] = Pb_thr[k] + (2-g)*Q[i,k]/(1-h)
							Qa_thr[i*K+k] = Qa_thr[i*K+k] + g*P[j,k]/h
							Qb_thr[i*K+k] = Qb_thr[i*K+k] + (2-g)*(1-P[j,k])/(1-h)
						a_thr[i] = a_thr[i] + 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break
			for k in range(K):
				Pa_thr[k] = Pa_thr[k]*P[j,k]
				Pb_thr[k] = Pb_thr[k]*(1-P[j,k])
				P[j,k] = Pa_thr[k]/(Pa_thr[k] + Pb_thr[k])
				P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))
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
cpdef void accelP(unsigned char[:,::1] G, double[:,::1] P, double[:,::1] Q, \
		double[:,::1] Qa, double[:,::1] Qb, double[:,::1] D, double[::1] a, int t):
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = P.shape[1]
		int i, j, k, x, y, i0, k0, b, bytepart
		double g, h, P0
		double* a_thr
		double* Pa_thr
		double* Pb_thr
		double* Qa_thr
		double* Qb_thr
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	with nogil, parallel(num_threads=t):
		a_thr = <double*>PyMem_RawMalloc(sizeof(double)*N)
		Pa_thr = <double*>PyMem_RawMalloc(sizeof(double)*K)
		Pb_thr = <double*>PyMem_RawMalloc(sizeof(double)*K)
		Qa_thr = <double*>PyMem_RawMalloc(sizeof(double)*N*K)
		Qb_thr = <double*>PyMem_RawMalloc(sizeof(double)*N*K)
		for i0 in range(N):
			a_thr[i0] = 0.0
			for k0 in range(K):
				Qa_thr[i0*K + k0] = 0.0
				Qb_thr[i0*K + k0] = 0.0
		for j in prange(M):
			for k in range(K):
				Pa_thr[k] = 0.0
				Pb_thr[k] = 0.0
			i = 0
			for b in range(B):
				byte = G[j,b]
				for bytepart in range(4):
					if recode[byte & mask] != 9:
						g = <double>recode[byte & mask]
						h = 0.0
						for k in range(K):
							h = h + Q[i,k]*P[j,k]
						for k in range(K):
							Pa_thr[k] = Pa_thr[k] + g*Q[i,k]/h
							Pb_thr[k] = Pb_thr[k] + (2-g)*Q[i,k]/(1-h)
							Qa_thr[i*K+k] = Qa_thr[i*K+k] + g*P[j,k]/h
							Qb_thr[i*K+k] = Qb_thr[i*K+k] + (2-g)*(1-P[j,k])/(1-h)
						a_thr[i] = a_thr[i] + 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break
			for k in range(K):
				P0 = P[j,k]
				Pa_thr[k] = Pa_thr[k]*P[j,k]
				Pb_thr[k] = Pb_thr[k]*(1-P[j,k])
				P[j,k] = Pa_thr[k]/(Pa_thr[k] + Pb_thr[k])
				P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))
				D[j,k] = P[j,k] - P0
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

# Update Q
cpdef void updateQ(double[:,::1] Q, double[:,::1] Qa, double[:,::1] Qb, \
		double[::1] a) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k
		double sumQ
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = (Qa[i,k]*Q[i,k] + Qb[i,k]*Q[i,k])/(2.0*a[i])
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			Qa[i,k] = 0.0
			Qb[i,k] = 0.0
			sumQ += Q[i,k]
		a[i] = 0.0
		# map2domain (normalize)
		for k in range(K):
			Q[i,k] = Q[i,k]/sumQ

# Update Q in acceleration
cpdef void accelQ(double[:,::1] Q, double[:,::1] Qa, double[:,::1] Qb, \
		double[:,::1] D, double[::1] a) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k
		double sumQ
		double* Q0 = <double*>PyMem_RawMalloc(sizeof(double)*K)
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q0[k] = Q[i,k]
			Q[i,k] = (Qa[i,k]*Q[i,k] + Qb[i,k]*Q[i,k])/(2.0*a[i])
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			Qa[i,k] = 0.0
			Qb[i,k] = 0.0
			sumQ += Q[i,k]
		a[i] = 0.0
		# map2domain (normalize)
		for k in range(K):
			Q[i,k] = Q[i,k]/sumQ
			D[i,k] = Q[i,k] - Q0[k]
	PyMem_RawFree(Q0)

# Accelerated jump for P (SQUAREM)
cpdef void alphaP(double[:,::1] P, double[:,::1] P0, double[:,::1] D1, \
		double[:,::1] D2, double[:,::1] D3, int t) nogil:
	cdef:
		int M = P.shape[0]
		int K = P.shape[1]
		int j, k
		double sum1 = 0.0
		double sum2 = 0.0
		double alpha
	for j in range(M):
		for k in range(K):
			D3[j,k] = D2[j,k] - D1[j,k]
			sum1 += D1[j,k]*D1[j,k]
			sum2 += D3[j,k]*D3[j,k]
	alpha = max(1.0, sqrt(sum1)/sqrt(sum2))
	for j in prange(M, num_threads=t):
		for k in range(K):
			P[j,k] = P0[j,k] + 2.0*alpha*D1[j,k] + alpha*alpha*D3[j,k]
			P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))

# Accelerated jump for Q (SQUAREM)
cpdef void alphaQ(double[:,::1] Q, double[:,::1] Q0, double[:,::1] D1, \
		double[:,::1] D2, double[:,::1] D3) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sumQ
		double alpha
		double sum1 = 0.0
		double sum2 = 0.0
	for i in range(N):
		for k in range(K):
			D3[i,k] = D2[i,k] - D1[i,k]
			sum1 += D1[i,k]*D1[i,k]
			sum2 += D3[i,k]*D3[i,k]
	alpha = max(1.0, sqrt(sum1)/sqrt(sum2))
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = Q0[i,k] + 2.0*alpha*D1[i,k] + alpha*alpha*D3[i,k]
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			sumQ += Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ
