# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from libc.math cimport log, log1p, sqrt
from libc.stdlib cimport calloc, free

##### fastmixture ######
# Inline functions
cdef inline double project(double s) noexcept nogil:
	return min(max(s, 1e-5), 1-(1e-5))

cdef inline double computeH(const double* p, const double* q, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double h = 0.0
	for k in range(K):
		h += p[k]*q[k]
	return h

# Expand data into full genotype matrix
cpdef void expandGeno(const unsigned char[:,::1] B, unsigned char[:,::1] G, \
		double[::1] Q_nrm) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t N_b = B.shape[1]
		size_t i, j, b, x, bytepart
		double* Q_len
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
	with nogil, parallel():
		Q_len = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			i = 0
			for b in range(N_b):
				byte = B[j,b]
				for bytepart in range(4):
					G[j,i] = recode[byte & mask]
					if G[j,i] != 9:
						Q_len[i] += 1.0
					byte = byte >> 2
					i = i + 1
					if i == N:
						break
		with gil:
			for x in range(N):
				Q_nrm[x] += Q_len[x]
		free(Q_len)

# Initialize P in supervised mode
cpdef void initP(const unsigned char[:,::1] G, double[:,::1] P, \
		const unsigned char[::1] y) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = P.shape[1]
		size_t i, j, k
		double* x
		unsigned char D
	for j in prange(M):
		x = <double*>calloc(K, sizeof(double))
		for i in range(N):
			D = G[j,i]
			if D == 9:
				continue
			if y[i] > 0:
				P[j,y[i]-1] += <double>D
				x[y[i]-1] += 1.0
		for k in range(K):
			if x[k] > 0.0:
				P[j,k] /= (2.0*x[k])
			P[j,k] = project(P[j,k])
			x[k] = 0.0
		free(x)

# Initialize Q in supervised mode
cpdef void initQ(double[:,::1] Q, const unsigned char[::1] y) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double sumQ
	for i in range(N):
		if y[i] > 0:
			for k in range(K):
				if k == (y[i]-1):
					Q[i,k] = 1-(1e-5)
				else:
					Q[i,k] = 1e-5
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = project(Q[i,k])
			sumQ += Q[i,k]
		sumQ = 1.0/sumQ
		for k in range(K):
			Q[i,k] *= sumQ

# Update Q in supervised mode
cpdef void superQ(double[:,::1] Q, const unsigned char[::1] y) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double sumQ
	for i in range(N):
		if y[i] > 0:
			sumQ = 0.0
			for k in range(K):
				if k == (y[i]-1):
					Q[i,k] = 1-(1e-5)
				else:
					Q[i,k] = 1e-5
				sumQ += Q[i,k]
			sumQ = 1.0/sumQ
			for k in range(K):
				Q[i,k] *= sumQ

# Estimate minor allele frequencies
cpdef void estimateFreq(const unsigned char[:,::1] G, double[::1] f) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t i, j
		double n
		unsigned char D
	for j in prange(M):
		n = 0.0
		for i in range(N):
			D = G[j,i]
			if D == 9:
				continue
			f[j] += <double>D
			n = n + 1.0
		f[j] /= (2.0*n)

# Log-likelihood
cpdef double loglike(const unsigned char[:,::1] G, const double[:,::1] P, \
		const double[:,::1] Q) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, k
		double res = 0.0
		double g, h
		unsigned char D
	for j in prange(M):
		for i in range(N):
			D = G[j,i]
			if D == 9:
				continue
			g = <double>D
			h = computeH(&P[j,0], &Q[i,0], K)
			res += g*log(h) + (2.0-g)*log1p(-h)
	return res

# Root-mean-square error
cpdef double rmse(const double[:,::1] Q, const double[:,::1] Q_pre) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double r = 0.0
	for i in range(N):
		for k in range(K):
			r += (Q[i,k] - Q_pre[i,k])*(Q[i,k] - Q_pre[i,k])
	return sqrt(r/<double>(N*K))

# Sum-of-squares used in evaluation 
cpdef double sumSquare(const unsigned char[:,::1] G, const double[:,::1] P, \
		const double[:,::1] Q) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, k
		double res = 0.0
		double h, g
		unsigned char D
	for j in prange(M):
		for i in range(N):
			D = G[j,i]
			if D == 9:
				continue
			g = <double>D
			h = 2.0*computeH(&P[j,0], &Q[i,0], K)
			res += (g-h)*(g-h)
	return res

# Kullback-Leibler divergence with average for Jensen-Shannon
cpdef double divKL(const double[:,::1] A, const double[:,::1] B) noexcept nogil:
	cdef:
		size_t N = A.shape[0]
		size_t K = A.shape[1]
		size_t i, k
		double d = 0.0
		double a
	for i in range(N):
		for k in range(K):
			a = (A[i,k] + B[i,k])*0.5
			d += A[i,k]*log(A[i,k]/a + 1e-9)
	return d/<double>N
