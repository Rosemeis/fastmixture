# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.stdlib cimport calloc, free

##### fastmixture #####
### Inline functions
cdef inline double project(double s) noexcept nogil:
	return min(max(s, 1e-5), 1-(1e-5))

cdef inline double computeH(const double* p, const double* q, int K) noexcept nogil:
	cdef:
		int k
		double h = 0.0
	for k in range(K):
		h += p[k]*q[k]
	return h

cdef inline void innerJ(const double* p, const double* q, double* p_a, \
		double* p_b, double* q_thr, const double a, const double b, const int K) \
		noexcept nogil:
	cdef:
		int k
	for k in range(K):
		p_a[k] += q[k]*a
		p_b[k] += q[k]*b
		q_thr[k] += p[k]*a + (1.0 - p[k])*b

cdef inline void innerP(const double* q, double* p_a, double* p_b, \
		const double a, const double b, const int K) noexcept nogil:
	cdef:
		int k
	for k in range(K):
		p_a[k] += q[k]*a
		p_b[k] += q[k]*b

cdef inline void innerQ(const double* p, double* q_thr, \
		const double a, const double b, const int K) noexcept nogil:
	cdef:
		int k
	for k in range(K):
		q_thr[k] += p[k]*a + (1.0 - p[k])*b

cdef inline void outerP(double* p, double* p_a, double* p_b, const int K) \
		noexcept nogil:
	cdef:
		int k
	for k in range(K):
		p_a[k] *= p[k]
		p[k] = project(p_a[k]/(p_a[k] + p_b[k]*(1.0 - p[k])))
		p_a[k] = 0.0
		p_b[k] = 0.0

cdef inline void outerAccelP(const double* p, double* p_new, double* p_a, \
		double* p_b, const int K) noexcept nogil:
	cdef:
		int k
	for k in range(K):
		p_a[k] *= p[k]
		p_new[k] = project(p_a[k]/(p_a[k] + p_b[k]*(1.0 - p[k])))
		p_a[k] = 0.0
		p_b[k] = 0.0

cdef inline void outerQ(double* q, double* q_tmp, const double a, const int K) \
		noexcept nogil:
	cdef:
		int k
		double sumQ = 0.0
	for k in range(K):
		q[k] = project(q[k]*q_tmp[k]*a)
		q_tmp[k] = 0.0
		sumQ += q[k]
	for k in range(K):
		q[k] /= sumQ

cdef inline void outerAccelQ(const double* q, double* q_new, double* q_tmp, \
		const double a, const int K) noexcept nogil:
	cdef:
		int k
		double sumQ = 0.0
	for k in range(K):
		q_new[k] = project(q[k]*q_tmp[k]*a)
		q_tmp[k] = 0.0
		sumQ += q_new[k]
	for k in range(K):
		q_new[k] /= sumQ

cdef inline double computeC(const double* x0, const double* x1, const double* x2, \
		const int I, const int J) noexcept nogil:
	cdef:
		int i, j, k
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
	for i in range(I):
		for j in range(J):
			k = i*J + j
			u = x1[k]-x0[k]
			v = (x2[k]-x1[k])-u
			sum1 += u*u
			sum2 += u*v
	return -(sum1/sum2)

cdef inline double computeBatchC(const double* p0, const double* p1, const double* p2, \
		const long* s, const int I, const int J) noexcept nogil:
	cdef:
		int i, j, k, l
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
	for i in range(I):
		l = s[i]
		for j in range(J):
			k = l*J + j
			u = p1[k]-p0[k]
			v = (p2[k]-p1[k])-u
			sum1 += u*u
			sum2 += u*v
	return -(sum1/sum2)


### Update functions
# Update P and Q temp arrays
cpdef void updateP(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_tmp, const int t) \
		noexcept nogil:
	cdef:
		int M = G.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int i, j, k, x, y
		double a, b, g, h
		double* P_thr
		double* Q_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>calloc(2*K, sizeof(double))
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				g = <double>G[j,i]
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
		double[:,::1] P_new, const double[:,::1] Q, double[:,::1] Q_tmp, \
		const int t) noexcept nogil:
	cdef:
		int M = G.shape[0]
		int B = G.shape[1]
		int N = Q.shape[0]
		int K = P.shape[1]
		int i, j, k, x, y
		double a, b, g, h
		double* P_thr
		double* Q_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>calloc(2*K, sizeof(double))
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				g = <double>G[j,i]
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
cpdef void alphaP(double[:,::1] P0, const double[:,::1] P1, const double[:,::1] P2, \
		const int t) \
		noexcept nogil:
	cdef:
		int M = P0.shape[0]
		int K = P0.shape[1]
		int j, k
		double c1, c2
	c1 = min(max(computeC(&P0[0,0], &P1[0,0], &P2[0,0], M, K), 1.0), 256.0)
	c2 = 1.0 - c1
	for j in prange(M, num_threads=t):
		for k in range(K):
			P0[j,k] = project(c2*P1[j,k] + c1*P2[j,k])

# Update Q from temp arrays
cpdef void updateQ(double[:,::1] Q, double[:,::1] Q_tmp, const int M) \
		noexcept nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, j, k
		double a = 1.0/<double>(2*M)
	for i in range(N):
		outerQ(&Q[i,0], &Q_tmp[i,0], a, K)

# Update Q in acceleration
cpdef void accelQ(const double[:,::1] Q, double[:,::1] Q_new, double[:,::1] Q_tmp, \
		const int M) noexcept nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double a = 1.0/<double>(2*M)
	for i in range(N):
		outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], a, K)

# Accelerated jump for Q (QN)
cpdef void alphaQ(double[:,::1] Q0, const double[:,::1] Q1, const double[:,::1] Q2) \
		noexcept nogil:
	cdef:
		int N = Q0.shape[0]
		int K = Q0.shape[1]
		int i, k
		double c1, c2, sumQ
	c1 = min(max(computeC(&Q0[0,0], &Q1[0,0], &Q2[0,0], N, K), 1.0), 256.0)
	c2 = 1.0 - c1
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q0[i,k] = project(c2*Q1[i,k] + c1*Q2[i,k])
			sumQ += Q0[i,k]
		for k in range(K):
			Q0[i,k] /= sumQ	


### Batch functions
# Update P in acceleration
cpdef void accelBatchP(const unsigned char[:,::1] G, const double[:,::1] P, \
		double[:,::1] P_new, const double[:,::1] Q, double[:,::1] Q_tmp, \
		const long[::1] s, const int t) noexcept nogil:
	cdef:
		int M = s.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int i, j, k, l, x, y
		double a, b, g, h
		double* P_thr
		double* Q_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>calloc(2*K, sizeof(double))
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			l = s[j]
			for i in range(N):
				g = <double>G[l,i]
				h = computeH(&P[l,0], &Q[i,0], K)
				a = g/h
				b = (2.0-g)/(1.0-h)
				innerJ(&P[l,0], &Q[i,0], &P_thr[0], &P_thr[K], &Q_thr[i*K], a, b, K)
			outerAccelP(&P[l,0], &P_new[l,0], &P_thr[0], &P_thr[K], K)
		with gil:
			for x in range(N):
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(P_thr)
		free(Q_thr)

# Accelerated jump for P (QN)
cpdef void alphaBatchP(double[:,::1] P0, const double[:,::1] P1, \
		const double[:,::1] P2, const long[::1] s, const int t) noexcept nogil:
	cdef:
		int M = s.shape[0]
		int K = P0.shape[1]
		int j, k, l
		double sum1 = 0.0
		double sum2 = 0.0
		double c1, c2
	c1 = min(max(computeBatchC(&P0[0,0], &P1[0,0], &P2[0,0], &s[0], M, K), 1.0), 256.0)
	c2 = 1.0 - c1
	for j in prange(M, num_threads=t):
		l = s[j]
		for k in range(K):
			P0[l,k] = project(c2*P1[l,k] + c1*P2[l,k])

### Safety steps
# Update P
cpdef void stepP(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, const int t) \
		noexcept nogil:
	cdef:
		int M = G.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int i, j, k
		double a, b, g, h
		double* P_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>calloc(2*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				g = <double>G[j,i]
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
		double[:,::1] P_new, const double[:,::1] Q, const int t) \
		noexcept nogil:
	cdef:
		int M = G.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int i, j, k
		double a, b, g, h
		double* P_thr
	with nogil, parallel(num_threads=t):
		P_thr = <double*>calloc(2*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				g = <double>G[j,i]
				h = computeH(&P[j,0], &Q[i,0], K)
				a = g/h
				b = (2.0-g)/(1.0-h)
				innerP(&Q[i,0], &P_thr[0], &P_thr[K], a, b, K)
			outerAccelP(&P[j,0], &P_new[j,0], &P_thr[0], &P_thr[K], K)
		free(P_thr)

# Update Q temp arrays
cpdef void stepQ(const unsigned char[:,::1] G, double[:,::1] P, \
		const double[:,::1] Q, double[:,::1] Q_tmp, const int t) \
		noexcept nogil:
	cdef:
		int M = G.shape[0]
		int N = G.shape[1]
		int K = Q.shape[1]
		int i, j, k, x, y
		double a, b, g, h
		double* Q_thr
	with nogil, parallel(num_threads=t):
		Q_thr = <double*>calloc(N*K, sizeof(double))
		for j in prange(M):
			for i in range(N):
				g = <double>G[j,i]
				h = computeH(&P[j,0], &Q[i,0], K)
				a = g/h
				b = (2.0-g)/(1.0-h)
				innerQ(&P[j,0], &Q_thr[i*K], a, b, K)
		with gil:
			for x in range(N):
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(Q_thr)
