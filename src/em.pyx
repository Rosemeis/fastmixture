# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log, sqrt

##### fastmixture #####
# Estimate minor allele frequencies and initial log-likelihood
cpdef void estimateFreq(unsigned char[:,::1] G, double[::1] f, int N, int t) nogil:
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int i, j, k, x, y, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
		double h, g, n
	for j in prange(M, num_threads=t):
		i = 0
		n = 0.0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					g = <double>recode[byte & mask]
					n = n + 1.0
					f[j] += g
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
		f[j] /= (2.0*n)

# Update P and temp Q arrays
cpdef void updateP(unsigned char[:,::1] G, double[:,::1] P, double[:,::1] Q, \
		double[:,::1] sumQA, double[:,::1] sumQB, double[::1] a, int t):
	cdef:
		int M = G.shape[0]
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
				tmpQA[i0*K + k0] = 0.0
				tmpQB[i0*K + k0] = 0.0
		for j in prange(M):
			for k in range(K):
				sumAG[k] = 0.0
				sumBG[k] = 0.0
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
							sumAG[k] = sumAG[k] + g*Q[i,k]/h
							sumBG[k] = sumBG[k] + (2-g)*Q[i,k]/(1-h)
							tmpQA[i*K+k] = tmpQA[i*K+k] + g*P[j,k]/h
							tmpQB[i*K+k] = tmpQB[i*K+k] + (2-g)*(1-P[j,k])/(1-h)
						tmpA[i] = tmpA[i] + 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break
			for k in range(K):
				sumAG[k] = sumAG[k]*P[j,k]
				sumBG[k] = sumBG[k]*(1-P[j,k])
				P[j,k] = sumAG[k]/(sumAG[k] + sumBG[k])
				P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))
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

# Update Q
cpdef void updateQ(double[:,::1] Q, double[:,::1] sumQA, double[:,::1] sumQB, \
		double[::1] a) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k
		double sumQ
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = (sumQA[i,k]*Q[i,k] + sumQB[i,k]*Q[i,k])/(2.0*a[i])
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			sumQA[i,k] = 0.0
			sumQB[i,k] = 0.0
			sumQ = sumQ + Q[i,k]
		a[i] = 0.0
		# map2domain (normalize)
		for k in range(K):
			Q[i,k] = Q[i,k]/sumQ

# Update P in acceleration
cpdef void accelP(unsigned char[:,::1] G, double[:,::1] P, double[:,::1] Q, \
		double[:,::1] sumQA, double[:,::1] sumQB, double[:,::1] D, double[::1] a, \
		int t):
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = P.shape[1]
		int i, j, k, x, y, i0, k0, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
		double h, g, P_old
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
				tmpQA[i0*K + k0] = 0.0
				tmpQB[i0*K + k0] = 0.0
		for j in prange(M):
			for k in range(K):
				sumAG[k] = 0.0
				sumBG[k] = 0.0
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
							sumAG[k] = sumAG[k] + g*Q[i,k]/h
							sumBG[k] = sumBG[k] + (2-g)*Q[i,k]/(1-h)
							tmpQA[i*K+k] = tmpQA[i*K+k] + g*P[j,k]/h
							tmpQB[i*K+k] = tmpQB[i*K+k] + (2-g)*(1-P[j,k])/(1-h)
						tmpA[i] = tmpA[i] + 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break
			for k in range(K):
				P_old = P[j,k]
				sumAG[k] = sumAG[k]*P[j,k]
				sumBG[k] = sumBG[k]*(1 - P[j,k])
				P[j,k] = sumAG[k]/(sumAG[k] + sumBG[k])
				P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))
				D[j,k] = P[j,k] - P_old
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

# Update Q in acceleration
cpdef void accelQ(double[:,::1] Q, double[:,::1] sumQA, double[:,::1] sumQB, \
		double[:,::1] D, double[::1] a) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k
		double sumQ
		double* Q_tmp = <double*>PyMem_RawMalloc(sizeof(double)*K)
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q_tmp[k] = Q[i,k]
			Q[i,k] = (sumQA[i,k]*Q[i,k] + sumQB[i,k]*Q[i,k])/(2.0*a[i])
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			sumQA[i,k] = 0.0
			sumQB[i,k] = 0.0
			sumQ = sumQ + Q[i,k]
		a[i] = 0.0
		# map2domain (normalize)
		for k in range(K):
			Q[i,k] = Q[i,k]/sumQ
			D[i,k] = Q[i,k] - Q_tmp[k]
	PyMem_RawFree(Q_tmp)

# Compute step length for Q
cpdef double alphaQ(double[:,::1] D1, double[:,::1] D2, double[:,::1] D3) nogil:
	cdef:
		int N = D1.shape[0]
		int K = D1.shape[1]
		int i, k
		double a
		double sum1 = 0.0
		double sum2 = 0.0
	for i in range(N):
		for k in range(K):
			D3[i,k] = D2[i,k] - D1[i,k]
			sum1 += D1[i,k]*D1[i,k]
			sum2 += D3[i,k]*D3[i,k]
	a = max(1.0, sqrt(sum1/sum2))
	return a

# Compute step length for P
cpdef double alphaP(double[:,::1] D1, double[:,::1] D2, double[:,::1] D3, \
		double[::1] sP1, double[::1] sP2, int t) nogil:
	cdef:
		int M = D1.shape[0]
		int K = D1.shape[1]
		int i, j, k
		double a
		double sum1 = 0.0
		double sum2 = 0.0
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
	a = max(1.0, sqrt(sum1/sum2))
	return a

# Accelerated jump for P (SQUAREM)
cpdef void accelUpdateP(double[:,::1] P, double[:,::1] D1, double[:,::1] D3, \
		double alpha, int t) nogil:
	cdef:
		int M = P.shape[0]
		int K = P.shape[1]
		int j, k
	for j in prange(M, num_threads=t):
		for k in range(K):
			P[j,k] = P[j,k] + 2.0*alpha*D1[j,k] + alpha*alpha*D3[j,k]
			P[j,k] = min(max(P[j,k], 1e-5), 1-(1e-5))

# Accelerated jump for Q (SQUAREM)
cpdef void accelUpdateQ(double[:,::1] Q, double[:,::1] D1, double[:,::1] D3, \
		double alpha) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sumQ
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = Q[i,k] + 2.0*alpha*D1[i,k] + alpha*alpha*D3[i,k]
			Q[i,k] = min(max(Q[i,k], 1e-5), 1-(1e-5))
			sumQ = sumQ + Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ

# Log-likelihood
cpdef void loglike(unsigned char[:,::1] G, double[:,::1] P, double[:,::1] Q, \
		double[::1] lkVec, int t) nogil:
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k, b, bytepart
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
		double h, g
	for j in prange(M, num_threads=t):
		lkVec[j] = 0.0
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					g = <double>recode[byte & mask]
					h = 0.0
					for k in range(K):
						h = h + Q[i,k]*P[j,k]
					lkVec[j] += g*log(h) + (2-g)*log(1-h)
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
