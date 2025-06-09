# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport calloc, free

cdef double PRO_MIN = 1e-5
cdef double PRO_MAX = 1.0-(1e-5)
cdef double ACC_MIN = 1.0
cdef double ACC_MAX = 256.0

##### fastmixture #####
# Estimate individual allele frequencies
cdef inline double _computeH(
		const double* p, const double* q, const uint32_t K
	) noexcept nogil:
	cdef:
		double h = 0.0
		size_t k
	for k in range(K):
		h += p[k]*q[k]
	return h

# Inner loop updates for temp P and Q
cdef inline void _inner(
		const double* p, const double* q, double* p_a, double* p_b, double* q_t, const uint8_t g, const double h, 
		const uint32_t K
	) noexcept nogil:
	cdef:
		double d = <double>g
		double a = d/h
		double b = (2.0 - d)/(1.0 - h)
		double c = a - b
		double q_k
		size_t k
	for k in range(K):
		q_k = q[k]
		p_a[k] += q_k*a
		p_b[k] += q_k*b
		q_t[k] += p[k]*c + b

# Inner loop update for temp P
cdef inline void _innerP(
		const double* q, double* p_a, double* p_b, const uint8_t g, const double h, const uint32_t K
	) noexcept nogil:
	cdef:
		double d = <double>g
		double a = d/h
		double b = (2.0 - d)/(1.0 - h)
		double q_k
		size_t k
	for k in range(K):
		q_k = q[k]
		p_a[k] += q_k*a
		p_b[k] += q_k*b

# Inner loop update for temp Q
cdef inline void _innerQ(
		const double* p, double* q_t, const uint8_t g, const double h, const uint32_t K
	) noexcept nogil:
	cdef:
		double d = <double>g
		double a = d/h
		double b = (2.0 - d)/(1.0 - h)
		double c = a - b
		size_t k
	for k in range(K):
		q_t[k] += p[k]*c + b

# Outer loop update for P
cdef inline void _outerP(
		double* p, double* p_a, double* p_b, const uint32_t K
	) noexcept nogil:
	cdef:
		double a, b, c, d
		size_t k
	for k in range(K):
		c = p[k]
		a = p_a[k]
		b = p_b[k]
		d = a*c/(c*(a - b) + b)
		p[k] = PRO_MIN if d < PRO_MIN else (PRO_MAX if d > PRO_MAX else d)
		p_a[k] = 0.0
		p_b[k] = 0.0

# Outer loop accelerated update for P
cdef inline void _outerAccelP(
		const double* p, double* p_n, double* p_a, double* p_b, const uint32_t K
	) noexcept nogil:
	cdef:
		double a, b, c, d
		size_t k
	for k in range(K):
		c = p[k]
		a = p_a[k]
		b = p_b[k]
		d = a*c/(c*(a - b) + b)
		p_n[k] = PRO_MIN if d < PRO_MIN else (PRO_MAX if d > PRO_MAX else d)
		p_a[k] = 0.0
		p_b[k] = 0.0

# Outer loop update for Q
cdef inline void _outerQ(
		double* q, double* q_t, const double c, const uint32_t K
	) noexcept nogil:
	cdef:
		double sumQ = 0.0
		double a, tmpQ
		size_t k
	for k in range(K):
		a = q[k]*q_t[k]*c
		tmpQ = PRO_MIN if a < PRO_MIN else (PRO_MAX if a > PRO_MAX else a)
		sumQ += tmpQ
		q[k] = tmpQ
		q_t[k] = 0.0
	for k in range(K):
		q[k] /= sumQ

# Outer loop accelerated update for Q
cdef inline void _outerAccelQ(
		const double* q, double* q_n, double* q_t, const double c, const uint32_t K
	) noexcept nogil:
	cdef:
		double sumQ = 0.0
		double a, tmpQ
		size_t k
	for k in range(K):
		a = q[k]*q_t[k]*c
		tmpQ = PRO_MIN if a < PRO_MIN else (PRO_MAX if a > PRO_MAX else a)
		sumQ += tmpQ
		q_n[k] = tmpQ
		q_t[k] = 0.0
	for k in range(K):
		q_n[k] /= sumQ

# Estimate QN factor for P
cdef inline double _qnP(
		const double* p0, const double* p1, const double* p2, const uint32_t M, const uint32_t K
	) noexcept nogil:
	cdef:
		double sum1 = 0.0
		double sum2 = 0.0
		double c, p, u, v
		size_t j, k, l
	for j in prange(M):
		l = j*K
		for k in range(K):
			p = p1[l+k]
			u = p - p0[l+k]
			v = p2[l+k] - p - u
			sum1 += u*u
			sum2 += u*v
	c = -(sum1/sum2)
	return ACC_MIN if c < ACC_MIN else (ACC_MAX if c > ACC_MAX else c)

# Estimate QN factor for Q
cdef inline double _qnQ(
		const double* q0, const double* q1, const double* q2, const uint32_t N, const uint32_t K
	) noexcept nogil:
	cdef:
		double sum1 = 0.0
		double sum2 = 0.0
		double c, q, u, v
		size_t i, k, l
	for i in range(N):
		l = i*K
		for k in range(K):
			q = q1[l+k]
			u = q - q0[l+k]
			v = q2[l+k] - q - u
			sum1 += u*u
			sum2 += u*v
	c = -(sum1/sum2)
	return ACC_MIN if c < ACC_MIN else (ACC_MAX if c > ACC_MAX else c)

# Alpha update for P
cdef inline void _computeP(
		double* p0, const double* p1, const double* p2, const double c1, const double c2, const uint32_t K
	) noexcept nogil:
	cdef:
		double a
		size_t k
	for k in range(K):
		a = c2*p1[k] + c1*p2[k]
		p0[k] = PRO_MIN if a < PRO_MIN else (PRO_MAX if a > PRO_MAX else a)

# Alpha update for Q
cdef inline void _computeQ(
		double* q0, const double* q1, const double* q2, const double c1, const double c2, const uint32_t K
	) noexcept nogil:
	cdef:
		double sumQ = 0.0
		double a, tmpQ
		size_t k
	for k in range(K):
		a = c2*q1[k] + c1*q2[k]
		tmpQ = PRO_MIN if a < PRO_MIN else (PRO_MAX if a > PRO_MAX else a)
		sumQ += tmpQ
		q0[k] = tmpQ
	for k in range(K):
		q0[k] /= sumQ

# Estimate QN factor for batch P
cdef inline double _qnBatch(
		const double* p0, const double* p1, const double* p2, const uint32_t* s_var, const uint32_t M, const uint32_t K
	) noexcept nogil:
	cdef:
		double sum1 = 0.0
		double sum2 = 0.0
		double c, p, u, v
		size_t j, k, l
	for j in prange(M):
		l = s_var[j]*K
		for k in range(K):
			p = p1[l+k]
			u = p - p0[l+k]
			v = p2[l+k] - p - u
			sum1 += u*u
			sum2 += u*v
	c = -(sum1/sum2)
	return ACC_MIN if c < ACC_MIN else (ACC_MAX if c > ACC_MAX else c)


### Update functions
# Update P and Q temp arrays
cpdef void updateP(
		uint8_t[:,::1] G, double[:,::1] P, double[:,::1] Q, double[:,::1] Q_tmp
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		uint32_t K = Q.shape[1]
		double h
		double* p
		double* q
		double* p_thr
		double* q_thr
		size_t i, j, x, y
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		p_thr = <double*>calloc(2*K, sizeof(double))
		q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			g = &G[j,0]
			for i in range(N):
				if g[i] != 9:
					q = &Q[i,0]
					h = _computeH(p, q, K)
					_inner(p, q, &p_thr[0], &p_thr[K], &q_thr[i*K], g[i], h, K)
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
cpdef void accelP(
		uint8_t[:,::1] G, double[:,::1] P, double[:,::1] P_new, double[:,::1] Q, double[:,::1] Q_tmp
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		uint32_t K = Q.shape[1]
		double h
		double* p
		double* q
		double* p_thr
		double* q_thr
		size_t i, j, x, y
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		p_thr = <double*>calloc(2*K, sizeof(double))
		q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			g = &G[j,0]
			for i in range(N):
				if g[i] != 9:
					q = &Q[i,0]
					h = _computeH(p, q, K)
					_inner(p, q, &p_thr[0], &p_thr[K], &q_thr[i*K], g[i], h, K)
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
cpdef void alphaP(
		double[:,::1] P, const double[:,::1] P1, const double[:,::1] P2
	) noexcept nogil:
	cdef:
		uint32_t M = P.shape[0]
		uint32_t K = P.shape[1]
		double c1, c2
		size_t j
	c1 = _qnP(&P[0,0], &P1[0,0], &P2[0,0], M, K)
	c2 = 1.0 - c1
	for j in prange(M):
		_computeP(&P[j,0], &P1[j,0], &P2[j,0], c1, c2, K)

# Update Q from temp arrays
cpdef void updateQ(
		double[:,::1] Q, double[:,::1] Q_tmp, double[::1] q_nrm
	) noexcept nogil:
	cdef:
		uint32_t N = Q.shape[0]
		uint32_t K = Q.shape[1]
		size_t i
	for i in range(N):
		_outerQ(&Q[i,0], &Q_tmp[i,0], 1.0/(2.0*q_nrm[i]), K)

# Update Q in acceleration
cpdef void accelQ(
		const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, double[::1] q_nrm
	) noexcept nogil:
	cdef:
		uint32_t N = Q.shape[0]
		uint32_t K = Q.shape[1]
		size_t i
	for i in range(N):
		_outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], 1.0/(2.0*q_nrm[i]), K)

# Accelerated jump for Q (QN)
cpdef void alphaQ(
		double[:,::1] Q, const double[:,::1] Q1, const double[:,::1] Q2
	) noexcept nogil:
	cdef:
		uint32_t N = Q.shape[0]
		uint32_t K = Q.shape[1]
		size_t i
		double c1, c2
	c1 = _qnQ(&Q[0,0], &Q1[0,0], &Q2[0,0], N, K)
	c2 = 1.0 - c1
	for i in range(N):
		_computeQ(&Q[i,0], &Q1[i,0], &Q2[i,0], c1, c2, K)


### Batch functions
# Update P in batch acceleration
cpdef void accelBatchP(
		uint8_t[:,::1] G, double[:,::1] P, double[:,::1] P_new, double[:,::1] Q, double[:,::1] Q_tmp, 
		double[::1] q_bat, const uint32_t[::1] s_var
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = s_var.shape[0]
		uint32_t N = G.shape[1]
		uint32_t K = Q.shape[1]
		double h
		double* p
		double* q
		double* p_thr
		double* q_thr
		double* q_len
		size_t i, j, l, x, y
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		p_thr = <double*>calloc(2*K, sizeof(double))
		q_thr = <double*>calloc(N*K, sizeof(double))
		q_len = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			l = s_var[j]
			p = &P[l,0]
			g = &G[l,0]
			for i in range(N):
				if g[i] != 9:
					q_len[i] += 1.0
					q = &Q[i,0]
					h = _computeH(p, q, K)
					_inner(p, q, &p_thr[0], &p_thr[K], &q_thr[i*K], g[i], h, K)
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
cpdef void alphaBatchP(
		double[:,::1] P, const double[:,::1] P1, const double[:,::1] P2, const uint32_t[::1] s_var
	) noexcept nogil:
	cdef:
		uint32_t M = s_var.shape[0]
		uint32_t K = P.shape[1]
		double c1, c2
		size_t j, k, l
	c1 = _qnBatch(&P[0,0], &P1[0,0], &P2[0,0], &s_var[0], M, K)
	c2 = 1.0 - c1
	for j in prange(M):
		l = s_var[j]
		_computeP(&P[l,0], &P1[l,0], &P2[l,0], c1, c2, K)

# Batch update Q from temp arrays
cpdef void accelBatchQ(
		const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, double[::1] q_bat
	) noexcept nogil:
	cdef:
		uint32_t N = Q.shape[0]
		uint32_t K = Q.shape[1]
		size_t i
	for i in range(N):
		_outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], 1.0/(2.0*q_bat[i]), K)
		q_bat[i] = 0.0


### Safety steps
# Update P
cpdef void stepP(
		uint8_t[:,::1] G, double[:,::1] P, double[:,::1] Q
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		uint32_t K = Q.shape[1]
		double h
		double* p
		double* q
		double* p_thr
		size_t i, j
	with nogil, parallel():
		p_thr = <double*>calloc(2*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			g = &G[j,0]
			for i in range(N):
				if g[i] != 9:
					q = &Q[i,0]
					h = _computeH(p, q, K)
					_innerP(q, &p_thr[0], &p_thr[K], g[i], h, K)
			_outerP(p, &p_thr[0], &p_thr[K], K)
		free(p_thr)

# Update accelerated P
cpdef void stepAccelP(
		uint8_t[:,::1] G, double[:,::1] P, double[:,::1] P_new, double[:,::1] Q
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		uint32_t K = Q.shape[1]
		double h
		double* p
		double* q
		double* p_thr
		size_t i, j
	with nogil, parallel():
		p_thr = <double*>calloc(2*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			g = &G[j,0]
			for i in range(N):
				if g[i] != 9:
					q = &Q[i,0]
					h = _computeH(p, q, K)
					_innerP(q, &p_thr[0], &p_thr[K], g[i], h, K)
			_outerAccelP(p, &P_new[j,0], &p_thr[0], &p_thr[K], K)
		free(p_thr)

# Update Q temp arrays
cpdef void stepQ(
		uint8_t[:,::1] G, double[:,::1] P, const double[:,::1] Q, double[:,::1] Q_tmp
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = G.shape[0]
		uint32_t N = G.shape[1]
		uint32_t K = Q.shape[1]
		double h
		double* p
		double* q_thr
		size_t i, j, x, y
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			p = &P[j,0]
			g = &G[j,0]
			for i in range(N):
				if g[i] != 9:
					h = _computeH(p, &Q[i,0], K)
					_innerQ(p, &q_thr[i*K], g[i], h, K)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for x in range(N):
			for y in range(K):
				Q_tmp[x,y] += q_thr[x*K + y]
		omp.omp_unset_lock(&mutex)
		free(q_thr)
	omp.omp_destroy_lock(&mutex)

# Update Q temp arrays in batch acceleration
cpdef void stepBatchQ(
		uint8_t[:,::1] G, double[:,::1] P, const double[:,::1] Q, double[:,::1] Q_tmp, double[::1] q_bat,
		const uint32_t[::1] s_var
	) noexcept nogil:
	cdef:
		uint8_t* g
		uint32_t M = s_var.shape[0]
		uint32_t N = G.shape[1]
		uint32_t K = Q.shape[1]
		double h
		double* p
		double* q_thr
		double* q_len
		size_t i, j, l, x, y
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		q_thr = <double*>calloc(N*K, sizeof(double))
		q_len = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			l = s_var[j]
			p = &P[l,0]
			g = &G[l,0]
			for i in range(N):
				if g[i] != 9:
					q_len[i] += 1.0
					h = _computeH(p, &Q[i,0], K)
					_innerQ(p, &q_thr[i*K], g[i], h, K)
		
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
