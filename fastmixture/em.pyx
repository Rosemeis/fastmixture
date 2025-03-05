# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin
from libc.stdlib cimport calloc, free

##### fastmixture #####
### Inline functions
# Truncate parameters to domain
cdef inline double _project(const double s) noexcept nogil:
	cdef:
		double min_val = 1e-5
		double max_val = 1.0-(1e-5)
	return fmin(fmax(s, min_val), max_val)

# Estimate individual allele frequencies
cdef inline double _computeH(const double* p, const double* q, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double h = 0.0
	for k in range(K):
		h += p[k]*q[k]
	return h

# Inner loop updates for temp P and Q
cdef inline void _inner(const double* p, const double* q, double* p_a, double* p_b, \
		double* q_thr, const unsigned char g, const double h, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double d = <double>g
		double a = d/h
		double b = (2.0-d)/(1.0-h)
	for k in range(K):
		p_a[k] += q[k]*a
		p_b[k] += q[k]*b
		q_thr[k] += p[k]*(a - b) + b

# Inner loop update for temp P
cdef inline void _innerP(const double* q, double* p_a, double* p_b, \
		const unsigned char g, const double h, const size_t K) noexcept nogil:
	cdef:
		size_t k
		double d = <double>g
		double a = d/h
		double b = (2.0-d)/(1.0-h)
	for k in range(K):
		p_a[k] += q[k]*a
		p_b[k] += q[k]*b

# Inner loop update for temp Q
cdef inline void _innerQ(const double* p, double* q_thr, \
		const unsigned char g, const double h, const size_t K) noexcept nogil:
	cdef:
		size_t k
		double d = <double>g
		double a = d/h
		double b = (2.0-d)/(1.0-h)
	for k in range(K):
		q_thr[k] += p[k]*(a - b) + b

# Outer loop update for P
cdef inline void _outerP(double* p, double* p_a, double* p_b, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		p[k] = _project((p_a[k]*p[k])/(p[k]*(p_a[k] - p_b[k]) + p_b[k]))
		p_a[k] = 0.0
		p_b[k] = 0.0

# Outer loop accelerated update for P
cdef inline void _outerAccelP(const double* p, double* p_n, double* p_a, \
		double* p_b, const size_t K) noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		p_n[k] = _project((p_a[k]*p[k])/(p[k]*(p_a[k] - p_b[k]) + p_b[k]))
		p_a[k] = 0.0
		p_b[k] = 0.0

# Outer loop update for Q
cdef inline void _outerQ(double* q, double* q_tmp, const double a, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double sumQ = 0.0
	for k in range(K):
		q[k] = _project(q[k]*q_tmp[k]*a)
		sumQ += q[k]
	for k in range(K):
		q[k] /= sumQ
		q_tmp[k] = 0.0

# Outer loop accelerated update for Q
cdef inline void _outerAccelQ(const double* q, double* q_new, double* q_tmp, \
		const double a, const size_t K) noexcept nogil:
	cdef:
		size_t k
		double sumQ = 0.0
	for k in range(K):
		q_new[k] = _project(q[k]*q_tmp[k]*a)
		sumQ += q_new[k]
	for k in range(K):
		q_new[k] /= sumQ
		q_tmp[k] = 0.0

# Estimate QN factor
cdef inline double _computeC(const double* x0, const double* x1, const double* x2, \
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
cdef inline void _computeP(double* p0, const double* p1, const double* p2, \
		const double c1, const size_t I) noexcept nogil:
	cdef:
		size_t i
		double c2 = 1.0 - c1
	for i in prange(I):
		p0[i] = _project(c2*p1[i] + c1*p2[i])

# Alpha update for Q
cdef inline void _computeQ(double* q0, const double* q1, const double* q2, \
		const double c1, const double c2, const size_t K) noexcept nogil:
	cdef:
		size_t k
		double sumQ = 0.0
	for k in range(K):
		q0[k] = _project(c2*q1[k] + c1*q2[k])
		sumQ += q0[k]
	for k in range(K):
		q0[k] /= sumQ

# Estimate QN factor for batch P
cdef inline double _computeBatchC(const double* p0, const double* p1, const double* p2, \
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
		double h
		double* p
		double* p_thr
		double* q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		p_thr = <double*>calloc(2*K, sizeof(double))
		q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			for i in range(N):
				if G[j,i] != 9:
					h = _computeH(p, &Q[i,0], K)
					_inner(p, &Q[i,0], &p_thr[0], &p_thr[K], &q_thr[i*K], G[j,i], h, K)
			_outerP(p, &p_thr[0], &p_thr[K], K)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(p_thr)
		free(q_thr)
	omp.omp_destroy_lock(&mutex)

# Update P in acceleration
cpdef void accelP(const unsigned char[:,::1] G, double[:,::1] P, double[:,::1] P_new, \
		const double[:,::1] Q, double[:,::1] Q_tmp) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, x, y
		double h
		double* p
		double* p_thr
		double* q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		p_thr = <double*>calloc(2*K, sizeof(double))
		q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			for i in range(N):
				if G[j,i] != 9:
					h = _computeH(p, &Q[i,0], K)
					_inner(p, &Q[i,0], &p_thr[0], &p_thr[K], &q_thr[i*K], G[j,i], h, K)
			_outerAccelP(p, &P_new[j,0], &p_thr[0], &p_thr[K], K)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(p_thr)
		free(q_thr)
	omp.omp_destroy_lock(&mutex)

# Accelerated jump for P (QN)
cpdef void alphaP(double[:,::1] P, const double[:,::1] P1, const double[:,::1] P2) \
		noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t K = P.shape[1]
		double c
	c = _computeC(&P[0,0], &P1[0,0], &P2[0,0], M*K)
	_computeP(&P[0,0], &P1[0,0], &P2[0,0], c, M*K)

# Update Q from temp arrays
cpdef void updateQ(double[:,::1] Q, double[:,::1] Q_tmp, double[::1] q_nrm) \
		noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i
	for i in range(N):
		_outerQ(&Q[i,0], &Q_tmp[i,0], 1.0/(2.0*q_nrm[i]), K)

# Update Q in acceleration
cpdef void accelQ(const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, \
		double[::1] q_nrm) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i
	for i in range(N):
		_outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], 1.0/(2.0*q_nrm[i]), K)

# Accelerated jump for Q (QN)
cpdef void alphaQ(double[:,::1] Q, const double[:,::1] Q1, const double[:,::1] Q2) \
		noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i
		double c1, c2
	c1 = _computeC(&Q[0,0], &Q1[0,0], &Q2[0,0], N*K)
	c2 = 1.0 - c1
	for i in range(N):
		_computeQ(&Q[i,0], &Q1[i,0], &Q2[i,0], c1, c2, K)


### Batch functions
# Update P in batch acceleration
cpdef void accelBatchP(const unsigned char[:,::1] G, double[:,::1] P, \
		double[:,::1] P_new, const double[:,::1] Q, double[:,::1] Q_tmp, \
		double[::1] q_bat, const unsigned int[::1] s) noexcept nogil:
	cdef:
		size_t M = s.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, l, x, y
		double h
		double* p
		double* p_thr
		double* q_thr
		double* q_len
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		p_thr = <double*>calloc(2*K, sizeof(double))
		q_thr = <double*>calloc(N*K, sizeof(double))
		q_len = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			l = s[j]
			p = &P[l,0]
			for i in range(N):
				if G[l,i] != 9:
					q_len[i] += 1.0
					h = _computeH(p, &Q[i,0], K)
					_inner(p, &Q[i,0], &p_thr[0], &p_thr[K], &q_thr[i*K], G[l,i], h, K)
			_outerAccelP(p, &P_new[l,0], &p_thr[0], &p_thr[K], K)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			q_bat[x] += q_len[x]
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(p_thr)
		free(q_thr)
		free(q_len)
	omp.omp_destroy_lock(&mutex)

# Batch accelerated jump for P (QN)
cpdef void alphaBatchP(double[:,::1] P, const double[:,::1] P1, \
		const double[:,::1] P2, const unsigned int[::1] s) noexcept nogil:
	cdef:
		size_t M = s.shape[0]
		size_t K = P.shape[1]
		size_t j, k, l
		double c1, c2
	c1 = _computeBatchC(&P[0,0], &P1[0,0], &P2[0,0], &s[0], M, K)
	c2 = 1.0 - c1
	for j in prange(M):
		l = s[j]
		for k in range(K):
			P[l,k] = _project(c2*P1[l,k] + c1*P2[l,k])

# Batch update Q from temp arrays
cpdef void accelBatchQ(const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, \
		double[::1] q_bat) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i
	for i in range(N):
		_outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], 1.0/(2.0*q_bat[i]), K)
		q_bat[i] = 0.0


### Safety steps
# Update P
cpdef void stepP(const unsigned char[:,::1] G, double[:,::1] P, const double[:,::1] Q) \
		noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j
		double h
		double* p
		double* p_thr
	with nogil, parallel():
		p_thr = <double*>calloc(2*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			for i in range(N):
				if G[j,i] != 9:
					h = _computeH(p, &Q[i,0], K)
					_innerP(&Q[i,0], &p_thr[0], &p_thr[K], G[j,i], h, K)
			_outerP(p, &p_thr[0], &p_thr[K], K)
		free(p_thr)

# Update accelerated P
cpdef void stepAccelP(const unsigned char[:,::1] G, double[:,::1] P, \
		double[:,::1] P_new, const double[:,::1] Q) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j
		double h
		double* p
		double* p_thr
	with nogil, parallel():
		p_thr = <double*>calloc(2*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			for i in range(N):
				if G[j,i] != 9:
					h = _computeH(p, &Q[i,0], K)
					_innerP(&Q[i,0], &p_thr[0], &p_thr[K], G[j,i], h, K)
			_outerAccelP(p, &P_new[j,0], &p_thr[0], &p_thr[K], K)
		free(p_thr)

# Update Q temp arrays
cpdef void stepQ(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_tmp) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, x, y
		double h
		double* p
		double* q_thr
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			for i in range(N):
				if G[j,i] != 9:
					h = _computeH(p, &Q[i,0], K)
					_innerQ(p, &q_thr[i*K], G[j,i], h, K)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(q_thr)
	omp.omp_destroy_lock(&mutex)

# Update Q temp arrays in batch acceleration
cpdef void stepBatchQ(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_tmp, double[::1] q_bat, \
		const unsigned int[::1] s) noexcept nogil:
	cdef:
		size_t M = s.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, l, x, y
		double h
		double* p
		double* q_thr
		double* q_len
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		q_thr = <double*>calloc(N*K, sizeof(double))
		q_len = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			l = s[j]
			p = &P[l,0]
			for i in range(N):
				if G[l,i] != 9:
					q_len[i] += 1.0
					h = _computeH(p, &Q[i,0], K)
					_innerQ(p, &q_thr[i*K], G[l,i], h, K)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			q_bat[x] += q_len[x]
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(q_thr)
		free(q_len)
	omp.omp_destroy_lock(&mutex)
