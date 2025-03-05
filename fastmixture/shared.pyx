# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, log, log1p, sqrt
from libc.stdlib cimport calloc, free

##### fastmixture ######
# Inline function for truncating parameters to domain
cdef inline double _project(const double s) noexcept nogil:
	cdef:
		double min_val = 1e-5
		double max_val = 1.0-(1e-5)
	return fmin(fmax(s, min_val), max_val)

# Inline function for computing individual allele frequency
cdef inline double _computeH(const double* p, const double* q, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double h = 0.0
	for k in range(K):
		h += p[k]*q[k]
	return h

# Expand data from 2-bit to 8-bit genotype matrix
cpdef void expandGeno(const unsigned char[:,::1] B, unsigned char[:,::1] G, \
		double[::1] q_nrm) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t N_b = B.shape[1]
		size_t i, j, b, x, bit
		double* Q_cnt
		unsigned char[4] recode = [2, 9, 1, 0]
		unsigned char mask = 3
		unsigned char byte
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		Q_cnt = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			i = 0
			for b in range(N_b):
				byte = B[j,b]
				for bit in range(4):
					G[j,i] = recode[(byte >> 2*bit) & mask]
					if G[j,i] != 9:
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
cpdef void initP(const unsigned char[:,::1] G, double[:,::1] P, \
		const unsigned char[::1] y) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = P.shape[1]
		size_t i, j, k
		double* x
	for j in prange(M):
		x = <double*>calloc(K, sizeof(double))
		for i in range(N):
			if G[j,i] == 9:
				continue
			if y[i] > 0:
				P[j,y[i]-1] += <double>G[j,i]
				x[y[i]-1] += 1.0
		for k in range(K):
			if x[k] > 0.0:
				P[j,k] /= (2.0*x[k])
			P[j,k] = _project(P[j,k])
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
					Q[i,k] = 1.0-(1e-5)
				else:
					Q[i,k] = 1e-5
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = _project(Q[i,k])
			sumQ += Q[i,k]
		for k in range(K):
			Q[i,k] /= sumQ

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
					Q[i,k] = 1.0-(1e-5)
				else:
					Q[i,k] = 1e-5
				sumQ += Q[i,k]
			for k in range(K):
				Q[i,k] /= sumQ

# Estimate minor allele frequencies
cpdef void estimateFreq(const unsigned char[:,::1] G, float[::1] f) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t i, j
		float c, n
	for j in prange(M):
		c = 0.0
		n = 0.0
		for i in range(N):
			if G[j,i] != 9:
				c = c + <float>G[j,i]
				n = n + 1.0
		f[j] = c/(2.0*n)

# Log-likelihood
cpdef double loglike(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j
		double res = 0.0
		double d, h
		double* p
	for j in prange(M):
		p = &P[j,0]
		for i in range(N):
			if G[j,i] != 9:
				h = _computeH(p, &Q[i,0], K)
				d = <double>G[j,i]
				res += d*log(h) + (2.0-d)*log1p(-h)
	return res

# Root-mean-square error
cpdef double rmse(const double[:,::1] Q, const double[:,::1] Q_pre) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double res = 0.0
	for i in range(N):
		for k in range(K):
			res += (Q[i,k] - Q_pre[i,k])*(Q[i,k] - Q_pre[i,k])
	return sqrt(res/<double>(N*K))

# Sum-of-squares used in evaluation 
cpdef double sumSquare(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j
		double res = 0.0
		double d, h
		double* p
	for j in prange(M):
		p = &P[j,0]
		for i in range(N):
			if G[j,i] != 9:
				h = 2.0*_computeH(p, &Q[i,0], K)
				d = <double>G[j,i]
				res += (d-h)*(d-h)
	return res

# Kullback-Leibler divergence with average for Jensen-Shannon
cpdef double divKL(const double[:,::1] A, const double[:,::1] B) noexcept nogil:
	cdef:
		size_t N = A.shape[0]
		size_t K = A.shape[1]
		size_t i, k
		double eps = 1e-10
		double d = 0.0
		double a
	for i in range(N):
		for k in range(K):
			a = (A[i,k] + B[i,k])*0.5
			d += A[i,k]*log(A[i,k]/a + eps)
	return d/<double>N
