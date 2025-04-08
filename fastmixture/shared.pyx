# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, log, log1p, sqrt
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport calloc, free

cdef double PRO_MIN = 1e-5
cdef double PRO_MAX = 1.0-(1e-5)

##### fastmixture ######
# Inline function for truncating parameters to domain
cdef inline double _project(
		const double s
	) noexcept nogil:
	return fmin(fmax(s, PRO_MIN), PRO_MAX)

# Inline function for computing individual allele frequency
cdef inline double _computeH(
		const double* p, const double* q, const uint32_t K
	) noexcept nogil:
	cdef:
		size_t k
		double h = 0.0
	for k in range(K):
		h += p[k]*q[k]
	return h

# Expand data from 2-bit to 8-bit genotype matrix
cpdef void expandGeno(
		const uint8_t[:,::1] B, uint8_t[:,::1] G, double[::1] q_nrm
	) noexcept nogil:
	cdef:
		uint8_t[4] recode = [2, 9, 1, 0]
		uint8_t mask = 3
		uint8_t byte
		uint8_t* g
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		uint32_t N_b = B.shape[1]
		double* Q_cnt
		size_t i, j, b, x, bit
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		Q_cnt = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			i = 0
			g = &G[j,0]
			for b in range(N_b):
				byte = B[j,b]
				for bit in range(4):
					g[i] = recode[(byte >> 2*bit) & mask]
					if g[i] != 9:
						Q_cnt[i] += 1.0
					i = i + 1
					if i == N:
						break

		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			q_nrm[x] += Q_cnt[x]
		omp.omp_unset_lock(&mutex)
		free(Q_cnt)
	omp.omp_destroy_lock(&mutex)

# Initialize P in supervised mode
cpdef void initP(
		uint8_t[:,::1] G, double[:,::1] P, const uint8_t[::1] y
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		uint32_t K = P.shape[1]
		double* x
		size_t i, j, k
	for j in prange(M):
		x = <double*>calloc(K, sizeof(double))
		g = &G[j,0]
		for i in range(N):
			if g[i] == 9:
				continue
			if y[i] > 0:
				x[y[i] - 1] += 1.0
				P[j,y[i] - 1] += <double>g[i]
		for k in range(K):
			if x[k] > 0.0:
				P[j,k] /= (2.0*x[k])
			P[j,k] = _project(P[j,k])
			x[k] = 0.0
		free(x)

# Initialize Q in supervised mode
cpdef void initQ(
		double[:,::1] Q, const uint8_t[::1] y
	) noexcept nogil:
	cdef:
		uint32_t N = Q.shape[0]
		uint32_t K = Q.shape[1]
		double sumQ
		size_t i, k
	for i in range(N):
		if y[i] > 0:
			for k in range(K):
				if k == (y[i] - 1):
					Q[i,k] = PRO_MAX
				else:
					Q[i,k] = PRO_MIN
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = _project(Q[i,k])
			sumQ += Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ

# Update Q in supervised mode
cpdef void superQ(
		double[:,::1] Q, const uint8_t[::1] y
	) noexcept nogil:
	cdef:
		uint32_t N = Q.shape[0]
		uint32_t K = Q.shape[1]
		double sumQ
		size_t i, k
	for i in range(N):
		if y[i] > 0:
			sumQ = 0.0
			for k in range(K):
				if k == (y[i] - 1):
					Q[i,k] = PRO_MAX
				else:
					Q[i,k] = PRO_MIN
				sumQ += Q[i,k]
			for k in range(K):
				Q[i,k] /= sumQ

# Estimate minor allele frequencies
cpdef void estimateFreq(
		uint8_t[:,::1] G, float[::1] f
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		float c, n
		size_t i, j
	for j in prange(M):
		c = 0.0
		n = 0.0
		g = &G[j,0]
		for i in range(N):
			if g[i] != 9:
				c = c + <float>g[i]
				n = n + 1.0
		f[j] = c/(2.0*n)

# Log-likelihood
cpdef double loglike(
		uint8_t[:,::1] G, double[:,::1] P, const double[:,::1] Q
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		uint32_t K = Q.shape[1]
		double res = 0.0
		double d, h
		double* p
		size_t i, j
	for j in prange(M):
		p = &P[j,0]
		g = &G[j,0]
		for i in range(N):
			if g[i] != 9:
				h = _computeH(p, &Q[i,0], K)
				d = <double>g[i]
				res += d*log(h) + (2.0 - d)*log1p(-h)
	return res

# Root-mean-square error
cpdef double rmse(
		const double[:,::1] Q, const double[:,::1] Q_pre
	) noexcept nogil:
	cdef:
		uint32_t N = Q.shape[0]
		uint32_t K = Q.shape[1]
		double res = 0.0
		size_t i, k
	for i in range(N):
		for k in range(K):
			res += (Q[i,k] - Q_pre[i,k])*(Q[i,k] - Q_pre[i,k])
	return sqrt(res/<double>(N*K))

# Sum-of-squares used in evaluation 
cpdef double sumSquare(
		uint8_t[:,::1] G, double[:,::1] P, const double[:,::1] Q
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		uint32_t K = Q.shape[1]
		double res = 0.0
		double d, h
		double* p
		size_t i, j
	for j in prange(M):
		p = &P[j,0]
		g = &G[j,0]
		for i in range(N):
			if g[i] != 9:
				h = 2.0*_computeH(p, &Q[i,0], K)
				d = <double>g[i]
				res += (d - h)*(d - h)
	return res

# Kullback-Leibler divergence with average for Jensen-Shannon
cpdef double divKL(
		const double[:,::1] A, const double[:,::1] B
	) noexcept nogil:
	cdef:
		uint32_t N = A.shape[0]
		uint32_t K = A.shape[1]
		double eps = 1e-10
		double d = 0.0
		double a
		size_t i, k
	for i in range(N):
		for k in range(K):
			a = (A[i,k] + B[i,k])*0.5
			d += A[i,k]*log(A[i,k]/a + eps)
	return d/<double>N
