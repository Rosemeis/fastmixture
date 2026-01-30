# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport abort, calloc, free

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef double f64

cdef f64 PRO_MIN = 1e-5
cdef f64 PRO_MAX = 1.0 - (1e-5)
cdef f64 ACC_MIN = 1.0
cdef f64 ACC_MAX = 96.0
cdef inline f64 _clamp1(f64 a) noexcept nogil: return fmax(PRO_MIN, fmin(a, PRO_MAX))
cdef inline f64 _clamp2(f64 a) noexcept nogil: return fmax(ACC_MIN, fmin(a, ACC_MAX))

##### fastmixture - EM algorithm #####
# Estimate individual allele frequencies
cdef inline f64 _computeH(
        const f64* p, const f64* q, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 h = 0.0
    for k in range(K):
        h += p[k]*q[k]
    return h

# Inner loop updates for temp P and Q
cdef inline void _innerJ(
        const f64* p, const f64* q, f64* p_a, f64* p_b, f64* q_t, const f64 d, const f64 h, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 a = d/h
        f64 b = (2.0 - d)/(1.0 - h)
        f64 c = a - b
        f64 q_k
    for k in range(K):
        q_k = q[k]
        p_a[k] += q_k*a
        p_b[k] += q_k*b
        q_t[k] += p[k]*c + b

# Inner loop update for temp P
cdef inline void _innerP(
        const f64* q, f64* p_a, f64* p_b, const f64 d, const f64 h, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 a = d/h
        f64 b = (2.0 - d)/(1.0 - h)
        f64 q_k
    for k in range(K):
        q_k = q[k]
        p_a[k] += q_k*a
        p_b[k] += q_k*b

# Inner loop update for temp Q
cdef inline void _innerQ(
        const f64* p, f64* q_t, const f64 d, const f64 h, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 a = d/h
        f64 b = (2.0 - d)/(1.0 - h)
        f64 c = a - b
    for k in range(K):
        q_t[k] += p[k]*c + b

# Outer loop update for P
cdef inline void _outerP(
        f64* p, f64* p_a, f64* p_b, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 a, b, c, d
    for k in range(K):
        c = p[k]
        a = p_a[k]
        b = p_b[k]
        d = a*c/(c*(a - b) + b)
        p[k] = _clamp1(d)
        p_a[k] = 0.0
        p_b[k] = 0.0

# Outer loop accelerated update for P
cdef inline void _outerAccelP(
        const f64* p, f64* p_n, f64* p_a, f64* p_b, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 a, b, c, d
    for k in range(K):
        c = p[k]
        a = p_a[k]
        b = p_b[k]
        d = a*c/(c*(a - b) + b)
        p_n[k] = _clamp1(d)
        p_a[k] = 0.0
        p_b[k] = 0.0

# Outer loop update for Q
cdef inline void _outerQ(
        f64* q, f64* q_t, const f64 c, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 sumQ = 0.0
        f64 a, b
    for k in range(K):
        a = q[k]*q_t[k]*c
        b = _clamp1(a)
        sumQ += b
        q[k] = b
        q_t[k] = 0.0
    for k in range(K):
        q[k] /= sumQ

# Outer loop accelerated update for Q
cdef inline void _outerAccelQ(
        const f64* q, f64* q_n, f64* q_t, const f64 c, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 sumQ = 0.0
        f64 a, b
    for k in range(K):
        a = q[k]*q_t[k]*c
        b = _clamp1(a)
        sumQ += b
        q_n[k] = b
        q_t[k] = 0.0
    for k in range(K):
        q_n[k] /= sumQ

# Estimate QN factor for P
cdef inline f64 _qnP(
        f64* P0, f64* P1, f64* P2, const Py_ssize_t M, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t j, k, l
        f64 sum1 = 0.0
        f64 sum2 = 0.0
        f64 c, u, v
        f64* p0
        f64* p1
        f64* p2
    for j in prange(M, schedule='guided'):
        l = j*K
        p0 = &P0[l]
        p1 = &P1[l]
        p2 = &P2[l]
        for k in range(K):
            u = p1[k] - p0[k]
            v = p2[k] - p1[k] - u
            sum1 += u*u
            sum2 += u*v
    c = -(sum1/sum2)
    return _clamp2(c)

# Estimate QN factor for Q
cdef inline f64 _qnQ(
        const f64* q0, const f64* q1, const f64* q2, const Py_ssize_t I
    ) noexcept nogil:
    cdef:
        size_t i
        f64 sum1 = 0.0
        f64 sum2 = 0.0
        f64 c, u, v
    for i in range(I):
        u = q1[i] - q0[i]
        v = q2[i] - q1[i] - u
        sum1 += u*u
        sum2 += u*v
    c = -(sum1/sum2)
    return _clamp2(c)

# Estimate QN factor for batch P
cdef inline f64 _qnBatch(
        f64* P0, f64* P1, f64* P2, const u32* s_var, const Py_ssize_t M, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t j, k, l
        f64 sum1 = 0.0
        f64 sum2 = 0.0
        f64 c, u, v
        f64* p0
        f64* p1
        f64* p2
    for j in prange(M, schedule='guided'):
        l = s_var[j]*K
        p0 = &P0[l]
        p1 = &P1[l]
        p2 = &P2[l]
        for k in range(K):
            u = p1[k] - p0[k]
            v = p2[k] - p1[k] - u
            sum1 += u*u
            sum2 += u*v
    c = -(sum1/sum2)
    return _clamp2(c)

# Estimate QN factor for cross-validation Q
cdef inline f64 _qnCross(
        f64* Q0, f64* Q1, f64* Q2, const u32* s_ind, const Py_ssize_t N, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t i, k, l
        f64 sum1 = 0.0
        f64 sum2 = 0.0
        f64 c, u, v
        f64* q0
        f64* q1
        f64* q2
    for i in prange(N, schedule='guided'):
        l = s_ind[i]*K
        q0 = &Q0[l]
        q1 = &Q1[l]
        q2 = &Q2[l]
        for k in range(K):
            u = q1[k] - q0[k]
            v = q2[k] - q1[k] - u
            sum1 += u*u
            sum2 += u*v
    c = -(sum1/sum2)
    return _clamp2(c)

# QN jump update for P
cdef inline void _computeP(
        f64* p0, const f64* p1, const f64* p2, const f64 c1, const f64 c2, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 a
    for k in range(K):
        a = c2*p1[k] + c1*p2[k]
        p0[k] = _clamp1(a)

# QN jump update for Q
cdef inline void _computeQ(
        f64* q0, const f64* q1, const f64* q2, const f64 c1, const f64 c2, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 sumQ = 0.0
        f64 a, b
    for k in range(K):
        a = c2*q1[k] + c1*q2[k]
        b = _clamp1(a)
        sumQ += b
        q0[k] = b
    for k in range(K):
        q0[k] /= sumQ



### Update functions
# Update P and Q temporary arrays
cpdef void updateP(
        u8[:,::1] G, f64[:,::1] P, f64[:,::1] Q, f64[:,::1] Q_tmp
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        size_t i, j, x, y
        u8* g
        f64 h
        f64* p
        f64* q
        f64* p_thr
        f64* q_thr
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        # Thread-local buffer allocation
        p_thr = <f64*>calloc(2*K, sizeof(f64))
        if p_thr is NULL:
            abort()
        q_thr = <f64*>calloc(N*K, sizeof(f64))
        if q_thr is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            g = &G[j,0]
            p = &P[j,0]
            for i in range(N):
                if g[i] != 9:
                    q = &Q[i,0]
                    h = _computeH(p, q, K)
                    _innerJ(p, q, &p_thr[0], &p_thr[K], &q_thr[i*K], <f64>g[i], h, K)
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

# Update P and Q temporary arrays in acceleration
cpdef void accelP(
        u8[:,::1] G, f64[:,::1] P, f64[:,::1] P_new, f64[:,::1] Q, f64[:,::1] Q_tmp
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        size_t i, j, x, y
        u8* g
        f64 h
        f64* p
        f64* q
        f64* p_thr
        f64* q_thr
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        # Thread-local buffer allocation
        p_thr = <f64*>calloc(2*K, sizeof(f64))
        if p_thr is NULL:
            abort()
        q_thr = <f64*>calloc(N*K, sizeof(f64))
        if q_thr is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            g = &G[j,0]
            p = &P[j,0]
            for i in range(N):
                if g[i] != 9:
                    q = &Q[i,0]
                    h = _computeH(p, q, K)
                    _innerJ(p, q, &p_thr[0], &p_thr[K], &q_thr[i*K], <f64>g[i], h, K)
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
cpdef void jumpP(
        f64[:,::1] P, f64[:,::1] P1, f64[:,::1] P2
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = P.shape[0]
        Py_ssize_t K = P.shape[1]
        size_t j
        f64 c1, c2
    c1 = _qnP(&P[0,0], &P1[0,0], &P2[0,0], M, K)
    c2 = 1.0 - c1
    for j in prange(M, schedule='guided'):
        _computeP(&P[j,0], &P1[j,0], &P2[j,0], c1, c2, K)

# Update Q from temporary arrays
cpdef void updateQ(
        f64[:,::1] Q, f64[:,::1] Q_tmp, f64[::1] q_nrm
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i
    for i in range(N):
        _outerQ(&Q[i,0], &Q_tmp[i,0], 1.0/q_nrm[i], K)

# Update Q from temporary arrays in acceleration
cpdef void accelQ(
        const f64[:,::1] Q, f64[:,::1] Q_new, f64[:,::1] Q_tmp, f64[::1] q_nrm
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i
    for i in range(N):
        _outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], 1.0/q_nrm[i], K)

# Accelerated jump for Q (QN)
cpdef void jumpQ(
        f64[:,::1] Q, f64[:,::1] Q1, f64[:,::1] Q2
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i
        f64 c1, c2
    c1 = _qnQ(&Q[0,0], &Q1[0,0], &Q2[0,0], N*K)
    c2 = 1.0 - c1
    for i in range(N):
        _computeQ(&Q[i,0], &Q1[i,0], &Q2[i,0], c1, c2, K)



### Batch functions
# Update P in batch acceleration
cpdef void batchP(
        u8[:,::1] G, f64[:,::1] P, f64[:,::1] P_new, f64[:,::1] Q, f64[:,::1] Q_tmp, f64[::1] q_var, 
        const u32[::1] s_var
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = s_var.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        size_t i, j, l, x, y
        u8* g
        f64 h
        f64* p
        f64* q
        f64* p_thr
        f64* q_len
        f64* q_thr
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        # Thread-local buffer allocation
        p_thr = <f64*>calloc(2*K, sizeof(f64))
        if p_thr is NULL:
            abort()
        q_len = <f64*>calloc(N, sizeof(f64))
        if q_len is NULL:
            abort()
        q_thr = <f64*>calloc(N*K, sizeof(f64))
        if q_thr is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            l = s_var[j]
            g = &G[l,0]
            p = &P[l,0]
            for i in range(N):
                if g[i] != 9:
                    q_len[i] += 2.0
                    q = &Q[i,0]
                    h = _computeH(p, q, K)
                    _innerJ(p, q, &p_thr[0], &p_thr[K], &q_thr[i*K], <f64>g[i], h, K)
            _outerAccelP(p, &P_new[l,0], &p_thr[0], &p_thr[K], K)
        
        # omp critical
        omp.omp_set_lock(&mutex)
        for x in range(N):
            q_var[x] += q_len[x]
            for y in range(K):
                Q_tmp[x,y] += q_thr[x*K + y]
        omp.omp_unset_lock(&mutex)
        free(p_thr)
        free(q_thr)
        free(q_len)
    omp.omp_destroy_lock(&mutex)

# Batch accelerated jump for P (QN)
cpdef void jumpBatchP(
        f64[:,::1] P, f64[:,::1] P1, f64[:,::1] P2, const u32[::1] s_var
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = s_var.shape[0]
        Py_ssize_t K = P.shape[1]
        size_t j, l
        f64 c1, c2
    c1 = _qnBatch(&P[0,0], &P1[0,0], &P2[0,0], &s_var[0], M, K)
    c2 = 1.0 - c1
    for j in prange(M, schedule='guided'):
        l = s_var[j]
        _computeP(&P[l,0], &P1[l,0], &P2[l,0], c1, c2, K)

# Update Q from temporary arrays (batch)
cpdef void batchQ(
        const f64[:,::1] Q, f64[:,::1] Q_new, f64[:,::1] Q_tmp, f64[::1] q_var
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i
    for i in range(N):
        _outerAccelQ(&Q[i,0], &Q_new[i,0], &Q_tmp[i,0], 1.0/q_var[i], K)
        q_var[i] = 0.0


### Cross-validation steps
# Update P in cross-validation
cpdef void crossP(
        u8[:,::1] G, f64[:,::1] P, f64[:,::1] Q, f64[:,::1] Q_tmp, const u32[::1] s_ind
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = s_ind.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i, j, l, x, y, z
        u8* g
        f64 h
        f64* p
        f64* q
        f64* p_thr
        f64* q_thr
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        # Thread-local buffer allocation
        p_thr = <f64*>calloc(2*K, sizeof(f64))
        if p_thr is NULL:
            abort()
        q_thr = <f64*>calloc(N*K, sizeof(f64))
        if q_thr is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            g = &G[j,0]
            p = &P[j,0]
            for i in range(N):
                l = s_ind[i]
                if g[l] != 9:
                    q = &Q[l,0]
                    h = _computeH(p, q, K)
                    _innerJ(p, q, &p_thr[0], &p_thr[K], &q_thr[i*K], <f64>g[l], h, K)
            _outerP(p, &p_thr[0], &p_thr[K], K)
        
        # omp critical
        omp.omp_set_lock(&mutex)
        for x in range(N):
            z = s_ind[x]
            for y in range(K):
                Q_tmp[z,y] += q_thr[x*K + y]
        omp.omp_unset_lock(&mutex)
        free(p_thr)
        free(q_thr)
    omp.omp_destroy_lock(&mutex)

# Update P in cross-validation in acceleration
cpdef void crossAccelP(
        u8[:,::1] G, f64[:,::1] P, f64[:,::1] P_new, f64[:,::1] Q, f64[:,::1] Q_tmp, const u32[::1] s_ind
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = s_ind.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i, j, l, x, y, z
        u8* g
        f64 h
        f64* p
        f64* q
        f64* p_thr
        f64* q_thr
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        # Thread-local buffer allocation
        p_thr = <f64*>calloc(2*K, sizeof(f64))
        if p_thr is NULL:
            abort()
        q_thr = <f64*>calloc(N*K, sizeof(f64))
        if q_thr is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            g = &G[j,0]
            p = &P[j,0]
            for i in range(N):
                l = s_ind[i]
                if g[l] != 9:
                    q = &Q[l,0]
                    h = _computeH(p, q, K)
                    _innerJ(p, q, &p_thr[0], &p_thr[K], &q_thr[i*K], <f64>g[l], h, K)
            _outerAccelP(p, &P_new[j,0], &p_thr[0], &p_thr[K], K)
        
        # omp critical
        omp.omp_set_lock(&mutex)
        for x in range(N):
            z = s_ind[x]
            for y in range(K):
                Q_tmp[z,y] += q_thr[x*K + y]
        omp.omp_unset_lock(&mutex)
        free(p_thr)
        free(q_thr)
    omp.omp_destroy_lock(&mutex)

# Update P in cross-validation
cpdef void crossStepQ(
        u8[:,::1] G, f64[:,::1] P, f64[:,::1] Q, f64[:,::1] Q_tmp, const u32[::1] s_ind
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = s_ind.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i, j, l, x, y, z
        u8* g
        f64 h
        f64* p
        f64* q_thr
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        # Thread-local buffer allocation
        q_thr = <f64*>calloc(N*K, sizeof(f64))
        if q_thr is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            g = &G[j,0]
            p = &P[j,0]
            for i in range(N):
                l = s_ind[i]
                if g[l] != 9:
                    h = _computeH(p, &Q[l,0], K)
                    _innerQ(p, &q_thr[i*K], <f64>g[l], h, K)
        
        # omp critical
        omp.omp_set_lock(&mutex)
        for x in range(N):
            z = s_ind[x]
            for y in range(K):
                Q_tmp[z,y] += q_thr[x*K + y]
        omp.omp_unset_lock(&mutex)
        free(q_thr)
    omp.omp_destroy_lock(&mutex)

# Update Q from temporary arrays (cross-validation)
cpdef void crossQ(
        f64[:,::1] Q, f64[:,::1] Q_tmp, f64[::1] q_nrm, const u32[::1] s_ind
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = s_ind.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i, l
    for i in range(N):
        l = s_ind[i]
        _outerQ(&Q[l,0], &Q_tmp[l,0], 1.0/q_nrm[l], K)

# Update Q from temporary arrays in acceleration (cross-validation)
cpdef void crossAccelQ(
        const f64[:,::1] Q, f64[:,::1] Q_new, f64[:,::1] Q_tmp, f64[::1] q_nrm, const u32[::1] s_ind
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = s_ind.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i, l
    for i in range(N):
        l = s_ind[i]
        _outerAccelQ(&Q[l,0], &Q_new[l,0], &Q_tmp[l,0], 1.0/q_nrm[l], K)

# Cross-validation accelerated jump for Q (QN)
cpdef void jumpCrossQ(
        f64[:,::1] Q, f64[:,::1] Q1, f64[:,::1] Q2, const u32[::1] s_ind
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = s_ind.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i, l
        f64 c1, c2
    c1 = _qnCross(&Q[0,0], &Q1[0,0], &Q2[0,0], &s_ind[0], N, K)
    c2 = 1.0 - c1
    for i in range(N):
        l = s_ind[i]
        _computeQ(&Q[l,0], &Q1[l,0], &Q2[l,0], c1, c2, K)



### Safety steps
# Update P
cpdef void stepP(
        u8[:,::1] G, f64[:,::1] P, f64[:,::1] Q
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        size_t i, j
        u8* g
        f64 h
        f64* p
        f64* q
        f64* p_thr
    with nogil, parallel():
        # Thread-local buffer allocation
        p_thr = <f64*>calloc(2*K, sizeof(f64))
        if p_thr is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            g = &G[j,0]
            p = &P[j,0]
            for i in range(N):
                if g[i] != 9:
                    q = &Q[i,0]
                    h = _computeH(p, q, K)
                    _innerP(q, &p_thr[0], &p_thr[K], <f64>g[i], h, K)
            _outerP(p, &p_thr[0], &p_thr[K], K)
        free(p_thr)

# Update accelerated P
cpdef void stepAccelP(
        u8[:,::1] G, f64[:,::1] P, f64[:,::1] P_new, f64[:,::1] Q
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        size_t i, j
        u8* g
        f64 h
        f64* p
        f64* q
        f64* p_thr
    with nogil, parallel():
        # Thread-local buffer allocation
        p_thr = <f64*>calloc(2*K, sizeof(f64))
        if p_thr is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            p = &P[j,0]
            g = &G[j,0]
            for i in range(N):
                if g[i] != 9:
                    q = &Q[i,0]
                    h = _computeH(p, q, K)
                    _innerP(q, &p_thr[0], &p_thr[K], <f64>g[i], h, K)
            _outerAccelP(p, &P_new[j,0], &p_thr[0], &p_thr[K], K)
        free(p_thr)

# Update Q temp arrays
cpdef void stepQ(
        u8[:,::1] G, f64[:,::1] P, const f64[:,::1] Q, f64[:,::1] Q_tmp
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        size_t i, j, x, y
        u8* g
        f64 h
        f64* p
        f64* q_thr
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        # Thread-local buffer allocation
        q_thr = <f64*>calloc(N*K, sizeof(f64))
        if q_thr is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            g = &G[j,0]
            p = &P[j,0]
            for i in range(N):
                if g[i] != 9:
                    h = _computeH(p, &Q[i,0], K)
                    _innerQ(p, &q_thr[i*K], <f64>g[i], h, K)
        
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
        u8[:,::1] G, f64[:,::1] P, const f64[:,::1] Q, f64[:,::1] Q_tmp, f64[::1] q_var,
        const u32[::1] s_var
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = s_var.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        size_t i, j, l, x, y
        u8* g
        f64 h
        f64* p
        f64* q_len
        f64* q_thr
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        # Thread-local buffer allocation
        q_len = <f64*>calloc(N, sizeof(f64))
        if q_len is NULL:
            abort()
        q_thr = <f64*>calloc(N*K, sizeof(f64))
        if q_thr is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            l = s_var[j]
            g = &G[l,0]
            p = &P[l,0]
            for i in range(N):
                if g[i] != 9:
                    q_len[i] += 2.0
                    h = _computeH(p, &Q[i,0], K)
                    _innerQ(p, &q_thr[i*K], <f64>g[i], h, K)
        
        # omp critical
        omp.omp_set_lock(&mutex)
        for x in range(N):
            q_var[x] += q_len[x]
            for y in range(K):
                Q_tmp[x,y] += q_thr[x*K + y]
        omp.omp_unset_lock(&mutex)
        free(q_thr)
        free(q_len)
    omp.omp_destroy_lock(&mutex)
