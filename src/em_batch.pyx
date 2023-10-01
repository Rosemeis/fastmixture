# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log, sqrt

##### Stochastic EM algorithm #####
# Update P in acceleration
cpdef void accelP(unsigned char[:,::1] G, float[:,::1] P, float[:,::1] Q, \
		float[:,::1] Pa, float[:,::1] Pb, float[:,::1] Qa, float[:,::1] Qb, \
		float[:,::1] D, float[::1] a, long[::1] idx, int t):
	cdef:
		int M = idx.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = Q.shape[1]
		int d, i, j, k, x, y, i0, k0, b, bytepart
		float h, g, Pp
		float* a_local
		float* Qa_local
		float* Qb_local
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	with nogil, parallel(num_threads=t):
		a_local = <float*>PyMem_RawMalloc(sizeof(float)*N)
		Qa_local = <float*>PyMem_RawMalloc(sizeof(float)*N*K)
		Qb_local = <float*>PyMem_RawMalloc(sizeof(float)*N*K)
		for i0 in range(N):
			a_local[i0] = 0.0
			for k0 in range(K):
				Qa_local[i0*K+k0] = 0.0
				Qb_local[i0*K+k0] = 0.0
		for j in prange(M):
			d = idx[j]
			i = 0
			for b in range(B):
				byte = G[idx[j],b]
				for bytepart in range(4):
					if recode[byte & mask] != 9:
						g = <float>recode[byte & mask]
						h = 0.0
						for k in range(K):
							h = h + Q[i,k]*P[d,k]
						for k in range(K):
							Pa[d,k] += g*Q[i,k]/h
							Pb[d,k] += (2-g)*Q[i,k]/(1-h)
							Qa_local[i*K+k] += g*P[d,k]/h
							Qb_local[i*K+k] += (2-g)*(1-P[d,k])/(1-h)
						a_local[i] += 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break					
			for k in range(K):
				Pp = P[d,k]
				Pa[d,k] *= P[d,k]
				Pb[d,k] *= (1-P[d,k])
				P[d,k] = Pa[d,k]/(Pa[d,k] + Pb[d,k])
				P[d,k] = min(max(P[d,k], 1e-5), 1-(1e-5))
				Pa[d,k] = 0.0
				Pb[d,k] = 0.0
				D[d,k] = P[d,k] - Pp
		with gil:
			for x in range(N):
				a[x] += a_local[x]
				for y in range(K):
					Qa[x,y] += Qa_local[x*K + y]
					Qb[x,y] += Qb_local[x*K + y]
		PyMem_RawFree(a_local)
		PyMem_RawFree(Qa_local)
		PyMem_RawFree(Qb_local)

# Accelerated jump for P (SQUAREM)
cpdef void alphaP(float[:,::1] P, float[:,::1] D1, float[:,::1] D2, \
		float[:,::1] D3, float[::1] sP1, float[::1] sP2, long[::1] idx, \
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
		sP1[j] = 0.0
		sP2[j] = 0.0
		for k in range(K):
			D3[d,k] = D2[d,k] - D1[d,k]
			sP1[j] += D1[d,k]*D1[d,k]
			sP2[j] += D3[d,k]*D3[d,k]
	for k in range(M):
		sum1 += sP1[k]
		sum2 += sP2[k]
	alpha = max(1.0, sqrt(sum1/sum2))
	for j in prange(M, num_threads=t):
		d = idx[j]
		for k in range(K):
			P[d,k] = P[d,k] + 2.0*alpha*D1[d,k] + alpha*alpha*D3[d,k]
			P[d,k] = min(max(P[d,k], 1e-5), 1-(1e-5))
