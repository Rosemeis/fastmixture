# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log, sqrt

##### Stochastic EM algorithm #####
# Update P
cpdef void updateP(unsigned char[:,::1] G, float[:,::1] P, float[:,::1] Q, \
		float[:,::1] Qa, float[:,::1] Qb, float[::1] a, long[::1] idx, int t):
	cdef:
		int M = idx.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = Q.shape[1]
		int d, i, j, k, x, y, i0, k0, b, bytepart
		float g, h
		float* a_thr
		float* Pa_thr
		float* Pb_thr
		float* Qa_thr
		float* Qb_thr
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	with nogil, parallel(num_threads=t):
		a_thr = <float*>PyMem_RawMalloc(sizeof(float)*N)
		Pa_thr = <float*>PyMem_RawMalloc(sizeof(float)*K)
		Pb_thr = <float*>PyMem_RawMalloc(sizeof(float)*K)
		Qa_thr = <float*>PyMem_RawMalloc(sizeof(float)*N*K)
		Qb_thr = <float*>PyMem_RawMalloc(sizeof(float)*N*K)
		for i0 in range(N):
			a_thr[i0] = 0.0
			for k0 in range(K):
				Qa_thr[i0*K+k0] = 0.0
				Qb_thr[i0*K+k0] = 0.0
		for j in prange(M):
			for k in range(K):
				Pa_thr[k] = 0.0
				Pb_thr[k] = 0.0
			d = idx[j]
			i = 0
			for b in range(B):
				byte = G[d,b]
				for bytepart in range(4):
					if recode[byte & mask] != 9:
						g = <float>recode[byte & mask]
						h = 0.0
						for k in range(K):
							h = h + Q[i,k]*P[d,k]
						for k in range(K):
							Pa_thr[k] = Pa_thr[k] + g*Q[i,k]/h
							Pb_thr[k] = Pb_thr[k] + (2-g)*Q[i,k]/(1-h)
							Qa_thr[i*K+k] = Qa_thr[i*K+k] + g*P[d,k]/h
							Qb_thr[i*K+k] = Qb_thr[i*K+k] + (2-g)*(1-P[d,k])/(1-h)
						a_thr[i] = a_thr[i] + 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break					
			for k in range(K):
				Pa_thr[k] = Pa_thr[k]*P[d,k]
				Pb_thr[k] = Pb_thr[k]*(1-P[d,k])
				P[d,k] = Pa_thr[k]/(Pa_thr[k] + Pb_thr[k])
				P[d,k] = min(max(P[d,k], 1e-5), 1-(1e-5))
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
cpdef void accelP(unsigned char[:,::1] G, float[:,::1] P, float[:,::1] Q, \
		float[:,::1] Qa, float[:,::1] Qb, float[:,::1] D, float[::1] a, \
		long[::1] idx, int t):
	cdef:
		int M = idx.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = Q.shape[1]
		int d, i, j, k, x, y, i0, k0, b, bytepart
		float g, h, P0
		float* a_thr
		float* Pa_thr
		float* Pb_thr
		float* Qa_thr
		float* Qb_thr
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	with nogil, parallel(num_threads=t):
		a_thr = <float*>PyMem_RawMalloc(sizeof(float)*N)
		Pa_thr = <float*>PyMem_RawMalloc(sizeof(float)*K)
		Pb_thr = <float*>PyMem_RawMalloc(sizeof(float)*K)
		Qa_thr = <float*>PyMem_RawMalloc(sizeof(float)*N*K)
		Qb_thr = <float*>PyMem_RawMalloc(sizeof(float)*N*K)
		for i0 in range(N):
			a_thr[i0] = 0.0
			for k0 in range(K):
				Qa_thr[i0*K+k0] = 0.0
				Qb_thr[i0*K+k0] = 0.0
		for j in prange(M):
			for k in range(K):
				Pa_thr[k] = 0.0
				Pb_thr[k] = 0.0
			d = idx[j]
			i = 0
			for b in range(B):
				byte = G[d,b]
				for bytepart in range(4):
					if recode[byte & mask] != 9:
						g = <float>recode[byte & mask]
						h = 0.0
						for k in range(K):
							h = h + Q[i,k]*P[d,k]
						for k in range(K):
							Pa_thr[k] = Pa_thr[k] + g*Q[i,k]/h
							Pb_thr[k] = Pb_thr[k] + (2-g)*Q[i,k]/(1-h)
							Qa_thr[i*K+k] = Qa_thr[i*K+k] + g*P[d,k]/h
							Qb_thr[i*K+k] = Qb_thr[i*K+k] + (2-g)*(1-P[d,k])/(1-h)
						a_thr[i] = a_thr[i] + 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break	
			for k in range(K):
				P0 = P[d,k]
				Pa_thr[k] = Pa_thr[k]*P[d,k]
				Pb_thr[k] = Pb_thr[k]*(1-P[d,k])
				P[d,k] = Pa_thr[k]/(Pa_thr[k] + Pb_thr[k])
				P[d,k] = min(max(P[d,k], 1e-5), 1-(1e-5))
				D[d,k] = P[d,k] - P0
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
cpdef void alphaP(float[:,::1] P, float[:,::1] P0, float[:,::1] D1, float[:,::1] D2, \
		float[:,::1] D3, float[::1] pr, float[::1] pv, long[::1] idx, \
		int t) nogil:
	cdef:
		int M = idx.shape[0]
		int K = P.shape[1]
		int d, i, j, k
		float alpha
		float sum1 = 0.0
		float sum2 = 0.0
	for j in prange(M, num_threads=t):
		d = idx[j]
		pr[j] = 0.0
		pv[j] = 0.0
		for k in range(K):
			D3[d,k] = D2[d,k] - D1[d,k]
			pr[j] += D1[d,k]*D1[d,k]
			pv[j] += D3[d,k]*D3[d,k]
	for k in range(M):
		sum1 += pr[k]
		sum2 += pv[k]
	alpha = -max(1.0, sqrt(sum1/sum2))
	for j in prange(M, num_threads=t):
		d = idx[j]
		for k in range(K):
			P[d,k] = P0[d,k] - 2.0*alpha*D1[d,k] + alpha*alpha*D3[d,k]
			P[d,k] = min(max(P[d,k], 1e-5), 1-(1e-5))
