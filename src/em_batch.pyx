# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log, sqrt

##### Stochastic EM algorithm #####
# Update P and temp Q arrays
cpdef void updateP(unsigned char[:,::1] G, double[:,::1] P, double[:,::1] Q, \
		double[:,::1] sumQA, double[:,::1] sumQB, double[::1] a, long[::1] idx, int t):
	cdef:
		int M = idx.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = P.shape[1]
		int i, j, k, x, y, i0, k0, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
		double h, g
		double* tmpA
		double* sumAG
		double* sumBG
		double* tmpQA
		double* tmpQB
	with nogil, parallel(num_threads=t):
		tmpA = <double*>PyMem_RawMalloc(sizeof(double)*N)
		sumAG = <double*>PyMem_RawMalloc(sizeof(double)*K)
		sumBG = <double*>PyMem_RawMalloc(sizeof(double)*K)
		tmpQA = <double*>PyMem_RawMalloc(sizeof(double)*N*K)
		tmpQB = <double*>PyMem_RawMalloc(sizeof(double)*N*K)
		for i0 in range(N):
			tmpA[i0] = 0.0
			for k0 in range(K):
				tmpQA[i0*K+k0] = 0.0
				tmpQB[i0*K+k0] = 0.0
		for j in prange(M):
			for k in range(K):
				sumAG[k] = 0.0
				sumBG[k] = 0.0
			i = 0
			for b in range(B):
				byte = G[idx[j],b]
				for bytepart in range(4):
					if recode[byte & mask] != 9:
						g = <double>recode[byte & mask]
						h = 0.0
						for k in range(K):
							h = h + Q[i,k]*P[idx[j],k]
						for k in range(K):
							sumAG[k] = sumAG[k] + g*Q[i,k]/h
							sumBG[k] = sumBG[k] + (2-g)*Q[i,k]/(1-h)
							tmpQA[i*K+k] += g*P[idx[j],k]/h
							tmpQB[i*K+k] += (2-g)*(1-P[idx[j],k])/(1-h)
						tmpA[i] += 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break					
			for k in range(K):
				sumAG[k] *= P[idx[j],k]
				sumBG[k] *= (1-P[idx[j],k])
				P[idx[j],k] = sumAG[k]/(sumAG[k] + sumBG[k])
				P[idx[j],k] = min(max(P[idx[j],k], 1e-5), 1-(1e-5))
		with gil:
			for x in range(N):
				a[x] += tmpA[x]
				for y in range(K):
					sumQA[x,y] += tmpQA[x*K + y]
					sumQB[x,y] += tmpQB[x*K + y]
		PyMem_RawFree(tmpA)
		PyMem_RawFree(sumAG)
		PyMem_RawFree(sumBG)
		PyMem_RawFree(tmpQA)
		PyMem_RawFree(tmpQB)

# Update P in acceleration
cpdef void accelP(unsigned char[:,::1] G, double[:,::1] P, double[:,::1] Q, \
		double[:,::1] sumQA, double[:,::1] sumQB, double[:,::1] D, double[::1] a, \
		long[::1] idx, int t):
	cdef:
		int M = idx.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = P.shape[1]
		int i, j, k, x, y, i0, k0, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
		double h, g, P_old
		double* tmpA
		double* sumAG
		double* sumBG
		double* tmpQA
		double* tmpQB
	with nogil, parallel(num_threads=t):
		tmpA = <double*>PyMem_RawMalloc(sizeof(double)*N)
		sumAG = <double*>PyMem_RawMalloc(sizeof(double)*K)
		sumBG = <double*>PyMem_RawMalloc(sizeof(double)*K)
		tmpQA = <double*>PyMem_RawMalloc(sizeof(double)*N*K)
		tmpQB = <double*>PyMem_RawMalloc(sizeof(double)*N*K)
		for i0 in range(N):
			tmpA[i0] = 0.0
			for k0 in range(K):
				tmpQA[i0*K+k0] = 0.0
				tmpQB[i0*K+k0] = 0.0
		for j in prange(M):
			for k in range(K):
				sumAG[k] = 0.0
				sumBG[k] = 0.0
			i = 0
			for b in range(B):
				byte = G[idx[j],b]
				for bytepart in range(4):
					if recode[byte & mask] != 9:
						g = <double>recode[byte & mask]
						h = 0.0
						for k in range(K):
							h = h + Q[i,k]*P[idx[j],k]
						for k in range(K):
							sumAG[k] += g*Q[i,k]/h
							sumBG[k] += (2-g)*Q[i,k]/(1-h)
							tmpQA[i*K+k] += g*P[idx[j],k]/h
							tmpQB[i*K+k] += (2-g)*(1-P[idx[j],k])/(1-h)
						tmpA[i] += 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break					
			for k in range(K):
				P_old = P[idx[j],k]
				sumAG[k] *= P[idx[j],k]
				sumBG[k] *= (1-P[idx[j],k])
				P[idx[j],k] = sumAG[k]/(sumAG[k] + sumBG[k])
				P[idx[j],k] = min(max(P[idx[j],k], 1e-5), 1-(1e-5))
				D[idx[j],k] = P[idx[j],k] - P_old
		with gil:
			for x in range(N):
				a[x] += tmpA[x]
				for y in range(K):
					sumQA[x,y] += tmpQA[x*K + y]
					sumQB[x,y] += tmpQB[x*K + y]
		PyMem_RawFree(tmpA)
		PyMem_RawFree(sumAG)
		PyMem_RawFree(sumBG)
		PyMem_RawFree(tmpQA)
		PyMem_RawFree(tmpQB)

# Compute step length for P
cpdef double alphaP(double[:,::1] D1, double[:,::1] D2, double[:,::1] D3, \
		double[::1] sP1, double[::1] sP2, long[::1] idx, int t) nogil:
	cdef:
		int M = idx.shape[0]
		int K = D1.shape[1]
		int i, j, k
		double a
		double sum1 = 0.0
		double sum2 = 0.0
	for j in prange(M, num_threads=t):
		sP1[j] = 0.0
		sP2[j] = 0.0
		for k in range(K):
			D3[idx[j],k] = D2[idx[j],k] - D1[idx[j],k]
			sP1[j] += D1[idx[j],k]*D1[idx[j],k]
			sP2[j] += D3[idx[j],k]*D3[idx[j],k]
	for k in range(M):
		sum1 += sP1[k]
		sum2 += sP2[k]
	a = max(1.0, sqrt(sum1/sum2))
	return a

# Accelerated jump for P (SQUAREM)
cpdef void accelUpdateP(double[:,::1] P, double[:,::1] D1, double[:,::1] D3, \
		double alpha, long[::1] idx, int t) nogil:
	cdef:
		int M = idx.shape[0]
		int K = P.shape[1]
		int j, k
	for j in prange(M, num_threads=t):
		for k in range(K):
			P[idx[j],k] = P[idx[j],k] + 2.0*alpha*D1[idx[j],k] + \
				alpha*alpha*D3[idx[j],k]
			P[idx[j],k] = min(max(P[idx[j],k], 1e-5), 1-(1e-5))
