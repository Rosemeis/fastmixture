# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin
from libc.stdlib cimport calloc, free

##### fastmixture #####
### Inline functions
# Truncate parameters to domain
cdef inline double project(const double s) noexcept nogil:
	cdef:
		double min_val = 1e-5
		double max_val = 1.0-(1e-5)
	return fmin(fmax(s, min_val), max_val)

# Estimate individual allele frequencies
cdef inline double computeH(const double* p, const double* q, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double h = 0.0
	for k in range(K):
		h += p[k]*q[k]
	return h

# Inner loop updates for temp P and Q
cdef inline void inner(const double* p, const double* q, double* p_a, double* p_b, \
		double* q_thr, const double g, const double h, const size_t K) noexcept nogil:
	cdef:
		size_t k
		double a = g/h
		double b = (2.0-g)/(1.0-h)
	for k in range(K):
		p_a[k] += q[k]*a
		p_b[k] += q[k]*b
		q_thr[k] += p[k]*(a - b) + b

# Inner loop update for temp P
cdef inline void innerP(const double* q, double* p_a, double* p_b, \
		const double g, const double h, const size_t K) noexcept nogil:
	cdef:
		size_t k
		double a = g/h
		double b = (2.0-g)/(1.0-h)
	for k in range(K):
		p_a[k] += q[k]*a
		p_b[k] += q[k]*b

# Inner loop update for temp Q
cdef inline void innerQ(const double* p, double* q_thr, \
		const double g, const double h, const size_t K) noexcept nogil:
	cdef:
		size_t k
		double a = g/h
		double b = (2.0-g)/(1.0-h)
	for k in range(K):
		q_thr[k] += p[k]*(a - b) + b

# Outer loop update for P
cdef inline void outerP(double* p, double* p_a, double* p_b, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double pa, pb, pk
	for k in range(K):
		pa = p_a[k]
		pb = p_b[k]
		pk = p[k]
		p[k] = project((pa*pk)/(pk*(pa - pb) + pb))
		p_a[k] = 0.0
		p_b[k] = 0.0

# Outer loop accelerated update for P
cdef inline void outerAccelP(const double* p, double* p_new, double* p_a, \
		double* p_b, const size_t K) noexcept nogil:
	cdef:
		size_t k
		double pa, pb, pk
	for k in range(K):
		pa = p_a[k]
		pb = p_b[k]
		pk = p[k]
		p_new[k] = project((pa*pk)/(pk*(pa - pb) + pb))
		p_a[k] = 0.0
		p_b[k] = 0.0

# Outer loop update for Q
cdef inline void outerQ(double* q, double* q_tmp, const double a, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double sumQ = 0.0
		double valQ
	for k in range(K):
		valQ = project(q[k]*q_tmp[k]*a)
		sumQ += valQ
		q[k] = valQ
	for k in range(K):
		q[k] /= sumQ
		q_tmp[k] = 0.0

# Outer loop accelerated update for Q
cdef inline void outerAccelQ(const double* q, double* q_new, double* q_tmp, \
		const double a, const size_t K) noexcept nogil:
	cdef:
		size_t k
		double sumQ = 0.0
		double valQ
	for k in range(K):
		valQ = project(q[k]*q_tmp[k]*a)
		sumQ += valQ
		q_new[k] = valQ
	for k in range(K):
		q_new[k] /= sumQ
		q_tmp[k] = 0.0

# Estimate QN factor
cdef inline double computeC(const double* x0, const double* x1, const double* x2, \
		const size_t I) noexcept nogil:
	cdef:
		size_t i
		double min_val = 1.0
		double max_val = 256.0
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
	for i in prange(I):
		u = x1[i] - x0[i]
		v = x2[i] - x1[i] - u
		sum1 += u*u
		sum2 += u*v
	return fmin(fmax(-(sum1/sum2), min_val), max_val)

# Alpha update for P
cdef inline void computeA(double* p0, const double* p1, const double* p2, \
		const double c1, const size_t I) noexcept nogil:
	cdef:
		size_t i
		double c2 = 1.0 - c1
	for i in prange(I):
		p0[i] = project(c2*p1[i] + c1*p2[i])

# Estimate QN factor for batch P
cdef inline double computeBatchC(const double* p0, const double* p1, const double* p2, \
		const unsigned int* s, const size_t I, const size_t J) noexcept nogil:
	cdef:
		size_t i, j, k, l
		double min_val = 1.0
		double max_val = 256.0
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
	for i in prange(I):
		l = s[i]
		for j in range(J):
			k = l*J + j
			u = p1[k] - p0[k]
			v = p2[k] - p1[k] - u
			sum1 += u*u
			sum2 += u*v
	return fmin(fmax(-(sum1/sum2), min_val), max_val)


### Update functions
# Update P and Q temp arrays
cpdef void updateP(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_tmp) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, x, y
		double g, h
		double* p
		double* P_thr
		double* Q_thr
		unsigned char d
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		P_thr = <double*>calloc(2*K, sizeof(double))
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			for i in range(N):
				d = G[j,i]
				if d != 9:
					g = <double>d
					h = computeH(p, &Q[i,0], K)
					inner(p, &Q[i,0], &P_thr[0], &P_thr[K], &Q_thr[i*K], g, h, K)
			outerP(p, &P_thr[0], &P_thr[K], K)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += Q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(P_thr)
		free(Q_thr)
	omp.omp_destroy_lock(&mutex)

# Update P in acceleration
cpdef void accelP(const unsigned char[:,::1] G, double[:,::1] P, double[:,::1] P_new, \
		const double[:,::1] Q, double[:,::1] Q_tmp) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, x, y
		double g, h
		double* p
		double* P_thr
		double* Q_thr
		unsigned char d
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		P_thr = <double*>calloc(2*K, sizeof(double))
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			for i in range(N):
				d = G[j,i]
				if d != 9:
					g = <double>d
					h = computeH(p, &Q[i,0], K)
					inner(p, &Q[i,0], &P_thr[0], &P_thr[K], &Q_thr[i*K], g, h, K)
			outerAccelP(p, &P_new[j,0], &P_thr[0], &P_thr[K], K)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += Q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(P_thr)
		free(Q_thr)
	omp.omp_destroy_lock(&mutex)

# Accelerated jump for P (QN)
cpdef void alphaP(double[:,::1] P0, const double[:,::1] P1, const double[:,::1] P2) \
		noexcept nogil:
	cdef:
		size_t M = P0.shape[0]
		size_t K = P0.shape[1]
		double c
	c = computeC(&P0[0,0], &P1[0,0], &P2[0,0], M*K)
	computeA(&P0[0,0], &P1[0,0], &P2[0,0], c, M*K)

# Update Q from temp arrays
cpdef void updateQ(double[:,::1] Q, double[:,::1] Q_tmp, double[::1] Q_nrm) \
		noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i
	for i in range(N):
		outerQ(&Q[i,0], &Q_tmp[i,0], 1.0/(2.0*Q_nrm[i]), K)

# Update Q in acceleration
cpdef void accelQ(const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, \
		double[::1] Q_nrm) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i
	for i in range(N):
		outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], 1.0/(2.0*Q_nrm[i]), K)

# Accelerated jump for Q (QN)
cpdef void alphaQ(double[:,::1] Q0, const double[:,::1] Q1, const double[:,::1] Q2) \
		noexcept nogil:
	cdef:
		size_t N = Q0.shape[0]
		size_t K = Q0.shape[1]
		size_t i, k
		double c1, c2, sumQ, valQ
	c1 = computeC(&Q0[0,0], &Q1[0,0], &Q2[0,0], N*K)
	c2 = 1.0 - c1
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			valQ = project(c2*Q1[i,k] + c1*Q2[i,k])
			sumQ += valQ
			Q0[i,k] = valQ
		for k in range(K):
			Q0[i,k] /= sumQ	


### Batch functions
# Update P in batch acceleration
cpdef void accelBatchP(const unsigned char[:,::1] G, double[:,::1] P, \
		double[:,::1] P_new, const double[:,::1] Q, double[:,::1] Q_tmp, \
		double[::1] Q_bat, const unsigned int[::1] s) noexcept nogil:
	cdef:
		size_t M = s.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, l, x, y
		double g, h
		double* p
		double* P_thr
		double* Q_thr
		double* Q_len
		unsigned char d
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		P_thr = <double*>calloc(2*K, sizeof(double))
		Q_thr = <double*>calloc(N*K, sizeof(double))
		Q_len = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			l = s[j]
			p = &P[l,0]
			for i in range(N):
				d = G[l,i]
				if d != 9:
					Q_len[i] += 1.0
					g = <double>d
					h = computeH(p, &Q[i,0], K)
					inner(p, &Q[i,0], &P_thr[0], &P_thr[K], &Q_thr[i*K], g, h, K)
			outerAccelP(p, &P_new[l,0], &P_thr[0], &P_thr[K], K)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			Q_bat[x] += Q_len[x]
			for y in range(K):
				Q_tmp[x,y] += Q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(P_thr)
		free(Q_thr)
		free(Q_len)
	omp.omp_destroy_lock(&mutex)

# Batch accelerated jump for P (QN)
cpdef void alphaBatchP(double[:,::1] P0, const double[:,::1] P1, \
		const double[:,::1] P2, const unsigned int[::1] s) noexcept nogil:
	cdef:
		size_t M = s.shape[0]
		size_t K = P0.shape[1]
		size_t j, k, l
		double sum1 = 0.0
		double sum2 = 0.0
		double c1, c2
	c1 = computeBatchC(&P0[0,0], &P1[0,0], &P2[0,0], &s[0], M, K)
	c2 = 1.0 - c1
	for j in prange(M):
		l = s[j]
		for k in range(K):
			P0[l,k] = project(c2*P1[l,k] + c1*P2[l,k])

# Batch update Q from temp arrays
cpdef void accelBatchQ(const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, \
		double[::1] Q_bat) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double a
	for i in range(N):
		outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], 1.0/(2.0*Q_bat[i]), K)
		Q_bat[i] = 0.0

### Safety steps
# Update P
cpdef void stepP(const unsigned char[:,::1] G, double[:,::1] P, const double[:,::1] Q) \
		noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j
		double g, h
		double* p
		double* P_thr
		unsigned char d
	with nogil, parallel():
		P_thr = <double*>calloc(2*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			for i in range(N):
				d = G[j,i]
				if d != 9:
					g = <double>d
					h = computeH(p, &Q[i,0], K)
					innerP(&Q[i,0], &P_thr[0], &P_thr[K], g, h, K)
			outerP(p, &P_thr[0], &P_thr[K], K)
		free(P_thr)

# Update accelerated P
cpdef void stepAccelP(const unsigned char[:,::1] G, double[:,::1] P, \
		double[:,::1] P_new, const double[:,::1] Q) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j
		double g, h
		double* p
		double* P_thr
		unsigned char d
	with nogil, parallel():
		P_thr = <double*>calloc(2*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			for i in range(N):
				d = G[j,i]
				if d != 9:
					g = <double>d
					h = computeH(p, &Q[i,0], K)
					innerP(&Q[i,0], &P_thr[0], &P_thr[K], g, h, K)
			outerAccelP(p, &P_new[j,0], &P_thr[0], &P_thr[K], K)
		free(P_thr)

# Update Q temp arrays
cpdef void stepQ(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_tmp) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, x, y
		double g, h
		double* p
		double* Q_thr
		unsigned char d
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			for i in range(N):
				d = G[j,i]
				if d != 9:
					g = <double>d
					h = computeH(p, &Q[i,0], K)
					innerQ(p, &Q_thr[i*K], g, h, K)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += Q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(Q_thr)
	omp.omp_destroy_lock(&mutex)
