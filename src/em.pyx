# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log, sqrt

##### fastmixture #####
# Update P and temp Q arrays
cpdef void updateP(unsigned char[:,::1] G, float[:,::1] P, float[:,::1] Q, \
		float[:,::1] Pa, float[:,::1] Pb, float[:,::1] Qa, float[:,::1] Qb, \
		float[::1] a, int t):
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k, x, y, i0, k0, b, bytepart
		float h, g
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
				Qa_local[i0*K + k0] = 0.0
				Qb_local[i0*K + k0] = 0.0
		for j in prange(M):
			i = 0
			for b in range(B):
				byte = G[j,b]
				for bytepart in range(4):
					if recode[byte & mask] != 9:
						g = <float>recode[byte & mask]
						h = 0.0
						for k in range(K):
							h = h + Q[i,k]*P[j,k]
						for k in range(K):
							Pa[j,k] += g*Q[i,k]/h
							Pb[j,k] += (2-g)*Q[i,k]/(1-h)
							Qa_local[i*K+k] += g*P[j,k]/h
							Qb_local[i*K+k] += (2-g)*(1-P[j,k])/(1-h)
						a_local[i] += 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break
			for k in range(K):
				Pa[j,k] *= P[j,k]
				Pb[j,k] *= (1-P[j,k])
				P[j,k] = Pa[j,k]/(Pa[j,k] + Pb[j,k])
				P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))
				Pa[j,k] = 0.0
				Pb[j,k] = 0.0
		with gil:
			for x in range(N):
				a[x] += a_local[x]
				for y in range(K):
					Qa[x,y] += Qa_local[x*K + y]
					Qb[x,y] += Qb_local[x*K + y]
		PyMem_RawFree(a_local)
		PyMem_RawFree(Qa_local)
		PyMem_RawFree(Qb_local)

# Update P in acceleration
cpdef void accelP(unsigned char[:,::1] G, float[:,::1] P, float[:,::1] Q, \
		float[:,::1] Pa, float[:,::1] Pb, float[:,::1] Qa, float[:,::1] Qb, \
		float[:,::1] D, float[::1] a, \
		int t):
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = P.shape[1]
		int i, j, k, x, y, i0, k0, b, bytepart
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
				Qa_local[i0*K + k0] = 0.0
				Qb_local[i0*K + k0] = 0.0
		for j in prange(M):
			i = 0
			for b in range(B):
				byte = G[j,b]
				for bytepart in range(4):
					if recode[byte & mask] != 9:
						g = <float>recode[byte & mask]
						h = 0.0
						for k in range(K):
							h = h + Q[i,k]*P[j,k]
						for k in range(K):
							Pa[j,k] += g*Q[i,k]/h
							Pb[j,k] += (2-g)*Q[i,k]/(1-h)
							Qa_local[i*K+k] += g*P[j,k]/h
							Qb_local[i*K+k] += (2-g)*(1-P[j,k])/(1-h)
						a_local[i] += 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break
			for k in range(K):
				Pp = P[j,k]
				Pa[j,k] *= P[j,k]
				Pb[j,k] *= (1-P[j,k])
				P[j,k] = Pa[j,k]/(Pa[j,k] + Pb[j,k])
				P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))
				Pa[j,k] = 0.0
				Pb[j,k] = 0.0
				D[j,k] = P[j,k] - Pp
		with gil:
			for x in range(N):
				a[x] += a_local[x]
				for y in range(K):
					Qa[x,y] += Qa_local[x*K + y]
					Qb[x,y] += Qb_local[x*K + y]
		PyMem_RawFree(a_local)
		PyMem_RawFree(Qa_local)
		PyMem_RawFree(Qb_local)

# Update Q
cpdef void updateQ(float[:,::1] Q, float[:,::1] Qa, float[:,::1] Qb, \
		float[::1] a) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k
		float sumQ
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
cpdef void accelQ(float[:,::1] Q, float[:,::1] Qa, float[:,::1] Qb, \
		float[:,::1] D, float[::1] a) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k
		float sumQ
		float* Qp = <float*>PyMem_RawMalloc(sizeof(float)*K)
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Qp[k] = Q[i,k]
			Q[i,k] = (Qa[i,k]*Q[i,k] + Qb[i,k]*Q[i,k])/(2.0*a[i])
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			Qa[i,k] = 0.0
			Qb[i,k] = 0.0
			sumQ += Q[i,k]
		a[i] = 0.0
		# map2domain (normalize)
		for k in range(K):
			Q[i,k] = Q[i,k]/sumQ
			D[i,k] = Q[i,k] - Qp[k]
	PyMem_RawFree(Qp)

# Accelerated jump for P (SQUAREM)
cpdef void alphaP(float[:,::1] P, float[:,::1] D1, float[:,::1] D2, \
		float[:,::1] D3, float[::1] sP1, float[::1] sP2, int t) nogil:
	cdef:
		int M = P.shape[0]
		int K = P.shape[1]
		int i, j, k
		float alpha
		float sum1 = 0.0
		float sum2 = 0.0
	for j in prange(M, num_threads=t):
			sP1[j] = 0.0
			sP2[j] = 0.0
			for k in range(K):
				D3[j,k] = D2[j,k] - D1[j,k]
				sP1[j] += D1[j,k]*D1[j,k]
				sP2[j] += D3[j,k]*D3[j,k]
	for i in range(M):
		sum1 += sP1[i]
		sum2 += sP2[i]
	alpha = max(1.0, sqrt(sum1/sum2))
	for j in prange(M, num_threads=t):
		for k in range(K):
			P[j,k] += 2.0*alpha*D1[j,k] + alpha*alpha*D3[j,k]
			P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))

# Accelerated jump for Q (SQUAREM)
cpdef void alphaQ(float[:,::1] Q, float[:,::1] D1, float[:,::1] D2, \
		float[:,::1] D3) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		float alpha, sumQ
		float sum1 = 0.0
		float sum2 = 0.0
	for i in range(N):
		for k in range(K):
			D3[i,k] = D2[i,k] - D1[i,k]
			sum1 += D1[i,k]*D1[i,k]
			sum2 += D3[i,k]*D3[i,k]
	alpha = max(1.0, sqrt(sum1/sum2))
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] += 2.0*alpha*D1[i,k] + alpha*alpha*D3[i,k]
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			sumQ += Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ
