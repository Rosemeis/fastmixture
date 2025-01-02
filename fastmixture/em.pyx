# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from libc.stdlib cimport calloc, free

##### fastmixture #####
### Inline functions
cdef inline double project(const double s) noexcept nogil:
	return min(max(s, 1e-5), 1-(1e-5))

cdef inline double computeH(const double* p, const double* q, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double h = 0.0
	for k in range(K):
		h += p[k]*q[k]
	return h

cdef inline void innerJ(const double* p, const double* q, double* p_a, \
		double* p_b, double* q_thr, const double a, const double b, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		p_a[k] += q[k]*a
		p_b[k] += q[k]*b
		q_thr[k] += p[k]*a + (1.0 - p[k])*b

cdef inline void innerP(const double* q, double* p_a, double* p_b, \
		const double a, const double b, const size_t K) noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		p_a[k] += q[k]*a
		p_b[k] += q[k]*b

cdef inline void innerQ(const double* p, double* q_thr, \
		const double a, const double b, const size_t K) noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		q_thr[k] += p[k]*a + (1.0 - p[k])*b

cdef inline void outerP(double* p, double* p_a, double* p_b, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		p_a[k] *= p[k]
		p[k] = project(p_a[k]/(p_a[k] + p_b[k]*(1.0 - p[k])))
		p_a[k] = 0.0
		p_b[k] = 0.0

cdef inline void outerAccelP(const double* p, double* p_new, double* p_a, \
		double* p_b, const size_t K) noexcept nogil:
	cdef:
		size_t k
	for k in range(K):
		p_a[k] *= p[k]
		p_new[k] = project(p_a[k]/(p_a[k] + p_b[k]*(1.0 - p[k])))
		p_a[k] = 0.0
		p_b[k] = 0.0

cdef inline void outerQ(double* q, double* q_tmp, const double a, const size_t K) \
		noexcept nogil:
	cdef:
		size_t k
		double sumQ = 0.0
	for k in range(K):
		q[k] = project(q[k]*q_tmp[k]*a)
		q_tmp[k] = 0.0
		sumQ += q[k]
	sumQ = 1.0/sumQ
	for k in range(K):
		q[k] *= sumQ

cdef inline void outerAccelQ(const double* q, double* q_new, double* q_tmp, \
		const double a, const size_t K) noexcept nogil:
	cdef:
		size_t k
		double sumQ = 0.0
	for k in range(K):
		q_new[k] = project(q[k]*q_tmp[k]*a)
		q_tmp[k] = 0.0
		sumQ += q_new[k]
	sumQ = 1.0/sumQ
	for k in range(K):
		q_new[k] *= sumQ

cdef inline double computeC(const double* x0, const double* x1, const double* x2, \
		const size_t I) noexcept nogil:
	cdef:
		size_t i
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
	for i in prange(I):
		u = x1[i]-x0[i]
		v = x2[i]-x1[i]-u
		sum1 += u*u
		sum2 += u*v
	return min(max(-(sum1/sum2), 1.0), 256.0)

cdef inline double computeBatchC(const double* p0, const double* p1, const double* p2, \
		const unsigned int* s, const size_t I, const size_t J) noexcept nogil:
	cdef:
		size_t i, j, k, l
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
	for i in prange(I):
		l = s[i]
		for j in range(J):
			k = l*J + j
			u = p1[k]-p0[k]
			v = (p2[k]-p1[k])-u
			sum1 += u*u
			sum2 += u*v
	return min(max(-(sum1/sum2), 1.0), 256.0)


### Update functions
# Update P and Q temp arrays
cpdef void updateP(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_tmp) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, k, x, y
		double a, b, g, h
		double* P_thr
		double* Q_thr
		unsigned char D
	with nogil, parallel():
		P_thr = <double*>calloc(2*K, sizeof(double))
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				D = G[j,i]
				if D == 9:
					continue
				g = <double>D
				h = computeH(&P[j,0], &Q[i,0], K)
				a = g/h
				b = (2.0-g)/(1.0-h)
				innerJ(&P[j,0], &Q[i,0], &P_thr[0], &P_thr[K], &Q_thr[i*K], a, b, K)
			outerP(&P[j,0], &P_thr[0], &P_thr[K], K)
		with gil:
			for x in range(N):
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(P_thr)
		free(Q_thr)

# Update P in acceleration
cpdef void accelP(const unsigned char[:,::1] G, const double[:,::1] P, \
		double[:,::1] P_new, const double[:,::1] Q, double[:,::1] Q_tmp) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t B = G.shape[1]
		size_t N = Q.shape[0]
		size_t K = P.shape[1]
		size_t i, j, k, x, y
		double a, b, g, h
		double* P_thr
		double* Q_thr
		unsigned char D
	with nogil, parallel():
		P_thr = <double*>calloc(2*K, sizeof(double))
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				D = G[j,i]
				if D == 9:
					continue
				g = <double>D
				h = computeH(&P[j,0], &Q[i,0], K)
				a = g/h
				b = (2.0-g)/(1.0-h)
				innerJ(&P[j,0], &Q[i,0], &P_thr[0], &P_thr[K], &Q_thr[i*K], a, b, K)
			outerAccelP(&P[j,0], &P_new[j,0], &P_thr[0], &P_thr[K], K)
		with gil:
			for x in range(N):
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(P_thr)
		free(Q_thr)

# Accelerated jump for P (QN)
cpdef void alphaP(double[:,::1] P0, const double[:,::1] P1, const double[:,::1] P2) \
		noexcept nogil:
	cdef:
		size_t M = P0.shape[0]
		size_t K = P0.shape[1]
		size_t j, k
		double c1, c2
	c1 = computeC(&P0[0,0], &P1[0,0], &P2[0,0], M*K)
	c2 = 1.0 - c1
	for j in prange(M):
		for k in range(K):
			P0[j,k] = project(c2*P1[j,k] + c1*P2[j,k])

# Update Q from temp arrays
cpdef void updateQ(double[:,::1] Q, double[:,::1] Q_tmp, double[::1] Q_nrm) \
		noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, j, k
		double a
	for i in range(N):
		a = 1.0/(2.0*Q_nrm[i])
		outerQ(&Q[i,0], &Q_tmp[i,0], a, K)

# Update Q in acceleration
cpdef void accelQ(const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, \
		double[::1] Q_nrm) noexcept nogil:
	cdef:
		size_t N = Q.shape[0]
		size_t K = Q.shape[1]
		size_t i, k
		double a
	for i in range(N):
		a = 1.0/(2.0*Q_nrm[i])
		outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], a, K)

# Accelerated jump for Q (QN)
cpdef void alphaQ(double[:,::1] Q0, const double[:,::1] Q1, const double[:,::1] Q2) \
		noexcept nogil:
	cdef:
		size_t N = Q0.shape[0]
		size_t K = Q0.shape[1]
		size_t i, k
		double c1, c2, sumQ
	c1 = computeC(&Q0[0,0], &Q1[0,0], &Q2[0,0], N*K)
	c2 = 1.0 - c1
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q0[i,k] = project(c2*Q1[i,k] + c1*Q2[i,k])
			sumQ += Q0[i,k]
		sumQ = 1.0/sumQ
		for k in range(K):
			Q0[i,k] *= sumQ	


### Batch functions
# Update P in batch acceleration
cpdef void accelBatchP(const unsigned char[:,::1] G, const double[:,::1] P, \
		double[:,::1] P_new, const double[:,::1] Q, double[:,::1] Q_tmp, \
		double[::1] Q_bat, const unsigned int[::1] s) noexcept nogil:
	cdef:
		size_t M = s.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, k, l, x, y
		double a, b, g, h
		double* P_thr
		double* Q_thr
		double* Q_len
		unsigned char D
	with nogil, parallel():
		P_thr = <double*>calloc(2*K, sizeof(double))
		Q_thr = <double*>calloc(N*K, sizeof(double))
		Q_len = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			l = s[j]
			for i in range(N):
				D = G[l,i]
				if D == 9:
					continue
				Q_len[i] += 1.0
				g = <double>D
				h = computeH(&P[l,0], &Q[i,0], K)
				a = g/h
				b = (2.0-g)/(1.0-h)
				innerJ(&P[l,0], &Q[i,0], &P_thr[0], &P_thr[K], &Q_thr[i*K], a, b, K)
			outerAccelP(&P[l,0], &P_new[l,0], &P_thr[0], &P_thr[K], K)
		with gil:
			for x in range(N):
				Q_bat[x] += Q_len[x]
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(P_thr)
		free(Q_thr)
		free(Q_len)

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
		a = 1.0/(2.0*Q_bat[i])
		outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], a, K)
		Q_bat[i] = 0.0

### Safety steps
# Update P
cpdef void stepP(const unsigned char[:,::1] G, double[:,::1] P, const double[:,::1] Q) \
		noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, k
		double a, b, g, h
		double* P_thr
		unsigned char D
	with nogil, parallel():
		P_thr = <double*>calloc(2*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				D = G[j,i]
				if D == 9:
					continue
				g = <double>D
				h = computeH(&P[j,0], &Q[i,0], K)
				a = g/h
				b = (2.0-g)/(1.0-h)
				innerP(&Q[i,0], &P_thr[0], &P_thr[K], a, b, K)
				for k in range(K):
					P_thr[k] += Q[i,k]*a
					P_thr[K+k] += Q[i,k]*b
			outerP(&P[j,0], &P_thr[0], &P_thr[K], K)
		free(P_thr)

# Update accelerated P
cpdef void stepAccelP(const unsigned char[:,::1] G, const double[:,::1] P, \
		double[:,::1] P_new, const double[:,::1] Q) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, k
		double a, b, g, h
		double* P_thr
		unsigned char D
	with nogil, parallel():
		P_thr = <double*>calloc(2*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				D = G[j,i]
				if D == 9:
					continue
				g = <double>D
				h = computeH(&P[j,0], &Q[i,0], K)
				a = g/h
				b = (2.0-g)/(1.0-h)
				innerP(&Q[i,0], &P_thr[0], &P_thr[K], a, b, K)
			outerAccelP(&P[j,0], &P_new[j,0], &P_thr[0], &P_thr[K], K)
		free(P_thr)

# Update Q temp arrays
cpdef void stepQ(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_tmp) noexcept nogil:
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t K = Q.shape[1]
		size_t i, j, k, x, y
		double a, b, g, h
		double* Q_thr
		unsigned char D
	with nogil, parallel():
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				D = G[j,i]
				if D == 9:
					continue
				g = <double>D
				h = computeH(&P[j,0], &Q[i,0], K)
				a = g/h
				b = (2.0-g)/(1.0-h)
				innerQ(&P[j,0], &Q_thr[i*K], a, b, K)
		with gil:
			for x in range(N):
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(Q_thr)
