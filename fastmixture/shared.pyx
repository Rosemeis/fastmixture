# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log, sqrt

##### fastmixture ######
# Estimate minor allele frequencies
cpdef void estimateFreq(unsigned char[:,::1] G, double[::1] f, int N, int t) nogil:
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int i, j, b, bytepart
		double g, n
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		i = 0
		n = 0.0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					f[j] += <double>recode[byte & mask]
					n = n + 1.0
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
		f[j] /= (2.0*n)

# Log-likelihood
cpdef void loglike(unsigned char[:,::1] G, double[:,::1] P, double[:,::1] Q, \
		double[::1] lkVec, int t) nogil:
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k, b, bytepart
		double h, g
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
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

# Sum-of-squares used for evaluation 
cpdef void sumSquare(unsigned char[:,::1] G, double[:,::1] P, double[:,::1] Q, \
		double[::1] lsVec, int t) nogil:
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k, b, bytepart
		double h, g
		unsigned char[4] recode = [0, 9, 1, 2]
		unsigned char mask = 3
		unsigned char byte
	for j in prange(M, num_threads=t):
		lsVec[j] = 0.0
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				if recode[byte & mask] != 9:
					g = <double>recode[byte & mask]
					h = 0.0
					for k in range(K):
						h = h + Q[i,k]*P[j,k]
					lsVec[j] += (g - 2*h)*(g - 2*h)
				byte = byte >> 2
				i = i + 1
				if i == N:
					break
