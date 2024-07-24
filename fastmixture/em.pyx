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

cdef inline void innerP(const double* p, const double* q, double* p_thr, \
		double* q_thr, const double a, const double b, const int K) noexcept nogil:
	cdef:
		int k
	for k in range(K):
		p_thr[k] += q[k]*a
		p_thr[K+k] += q[k]*b
		q_thr[k] += p[k]*a + (1.0 - p[k])*b

cdef inline void outerP(double* p, double* p_thr, const int K) noexcept nogil:
	cdef:
		int k
	for k in range(K):
		p_thr[k] *= p[k]
		p[k] = project(p_thr[k]/(p_thr[k] + p_thr[K+k]*(1.0 - p[k])))
		p_thr[k] = 0.0
		p_thr[K+k] = 0.0

cdef inline void outerAccelP(const double* p, double* p_new, double* p_thr, \
		const int K) noexcept nogil:
	cdef:
		int k
	for k in range(K):
		p_thr[k] *= p[k]
		p_new[k] = project(p_thr[k]/(p_thr[k] + p_thr[K+k]*(1.0 - p[k])))
		p_thr[k] = 0.0
		p_thr[K+k] = 0.0

cdef inline void outerQ(double* q, double* q_tmp, const double a, const int K) \
		noexcept nogil:
	cdef:
		int k
		double sumQ = 0.0
	for k in range(K):
		q[k] = project(q[k]*q_tmp[k]*a)
		sumQ += q[k]
	for k in range(K):
		q[k] /= sumQ
		q_tmp[k] = 0.0

cdef inline void outerAccelQ(double* q, double* q_new, double* q_tmp, const double a, \
		const int K) noexcept nogil:
	cdef:
		int k
		double sumQ = 0.0
	for k in range(K):
		q_new[k] = project(q[k]*q_tmp[k]*a)
		sumQ += q_new[k]
	for k in range(K):
		q_new[k] /= sumQ
		q_tmp[k] = 0.0

cdef inline double computeC(const double* p0, const double* p1, const double* p2, \
		const int I, const int J) noexcept nogil:
	cdef:
		int i, j
		double sum1 = 0.0
		double sum2 = 0.0
		double u
	for i in range(I):
		for j in range(J):
			u = p1[i*J + j] - p0[i*J + j]
			sum1 += u*u
			sum2 += u*((p2[i*J + j] - p1[i*J + j]) - u)
	return -(sum1/sum2)

cdef inline double computeBatchC(const double* p0, const double* p1, const double* p2, \
		const long* s, const int I, const int J) noexcept nogil:
	cdef:
		int i, j, l
		double sum1 = 0.0
		double sum2 = 0.0
		double u
	for i in range(I):
		l = s[i]
		for j in range(J):
			u = p1[l*J + j] - p0[l*J + j]
			sum1 += u*u
			sum2 += u*((p2[l*J + j] - p1[l*J + j]) - u)
	return -(sum1/sum2)


### Update functions
# Update P and temp Q arrays
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
				innerP(&P[j,0], &Q[i,0], &P_thr[0], &Q_thr[i*K], a, b, K)
			outerP(&P[j,0], &P_thr[0], K)
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
				innerP(&P[j,0], &Q[i,0], &P_thr[0], &Q_thr[i*K], a, b, K)
			outerAccelP(&P[j,0], &P_new[j,0], &P_thr[0], K)
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
		double sum1 = 0.0
		double sum2 = 0.0
		double c1, c2
	c1 = computeC(&P0[0,0], &P1[0,0], &P2[0,0], M, K)
	c2 = 1.0 - c1
	for j in prange(M, num_threads=t):
		for k in range(K):
			P0[j,k] = project(c2*P1[j,k] + c1*P2[j,k])

# Update Q
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
	c1 = computeC(&Q0[0,0], &Q1[0,0], &Q2[0,0], N, K)
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
cpdef void batchP(const unsigned char[:,::1] G, double[:,::1] P, \
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
				innerP(&P[l,0], &Q[i,0], &P_thr[0], &Q_thr[i*K], a, b, K)
			outerAccelP(&P[l,0], &P_new[l,0], &P_thr[0], K)
		with gil:
			for x in range(N):
				for y in range(K):
					Q_tmp[x,y] += Q_thr[x*K + y]
		free(P_thr)
		free(Q_thr)

# Accelerated jump for P (QN)
cpdef void alphaBatchP(double[:,::1] P0, const double[:,::1] P1, const double[:,::1] P2, \
		const long[::1] s, const int t) noexcept nogil:
	cdef:
		int M = s.shape[0]
		int K = P0.shape[1]
		int j, k, l
		double sum1 = 0.0
		double sum2 = 0.0
		double c1, c2
	c1 = computeBatchC(&P0[0,0], &P1[0,0], &P2[0,0], &s[0], M, K)
	c2 = 1.0 - c1
	for j in prange(M, num_threads=t):
		l = s[j]
		for k in range(K):
			P0[l,k] = project(c2*P1[l,k] + c1*P2[l,k])
