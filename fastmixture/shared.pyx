# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, log, sqrt
from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport abort, calloc, free

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef float f32
ctypedef double f64

cdef f64 PRO_MIN = 1e-5
cdef f64 PRO_MAX = 1.0 - (1e-5)
cdef f64 DEV_MIN = 1e-10
cdef inline f64 _clamp1(f64 a) noexcept nogil: return fmax(PRO_MIN, fmin(a, PRO_MAX))
cdef inline f64 _clamp2(f64 a) noexcept nogil: return fmax(DEV_MIN, a)


##### fastmixture - misc. functions ######
# Set up initialization of Q
cdef inline void _begQ(
        f64* q, const u8 y, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 sumQ = 0.0
        f64 a
    if y > 0:
        for k in range(k):
            q[k] = PRO_MAX if k == (y - 1) else PRO_MIN
    for k in range(K):
        a = q[k]
        q[k] = _clamp1(a)
        sumQ += q[k]
    for k in range(K):
        q[k] /= sumQ

# Set up supervised Q
cdef inline void _setQ(
        f64* q, const u8 y, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 sumQ = 0.0
    for k in range(K):
        q[k] = PRO_MAX if k == y else PRO_MIN
        sumQ += q[k]
    for k in range(K):
        q[k] /= sumQ

# Compute individual allele frequency
cdef inline void _computeH(
        f64* Q, const f64* p, f64* h, const Py_ssize_t N, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t i, k
        f64* q
    for i in range(N):
        q = &Q[i*K]
        for k in range(K):
            h[i] += p[k]*q[k]

# Estimate individual allele frequencies (older)
cdef inline f64 _computeI(
        const f64* p, const f64* q, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f64 h = 0.0
    for k in range(K):
        h += p[k]*q[k]
    return h

# Compute log-likelihood contribution
cdef inline f64 _computeL(
        const u8* g, f64* h, const Py_ssize_t N
    ) noexcept nogil:
    cdef:
        size_t i
        f64 r = 0.0
        f64 d
    for i in range(N):
        d = <f64>g[i]
        r += d*log(h[i]) + (2.0 - d)*log(1.0 - h[i])
        h[i] = 0.0
    return r

# Compute log-likelihood contribution (with missingness check)
cdef inline f64 _computeM(
        const u8* g, f64* h, const Py_ssize_t N
    ) noexcept nogil:
    cdef:
        size_t i
        f64 r = 0.0
        f64 d
    for i in range(N):
        d = <f64>g[i]
        r += d*log(h[i]) + (2.0 - d)*log(1.0 - h[i]) if g[i] != 9 else 0.0
        h[i] = 0.0
    return r

# Compute the squared difference
cdef inline f64 _computeR(
        const f64* a, const f64* b, const Py_ssize_t I
    ) noexcept nogil:
    cdef:
        size_t i
        f64 r = 0.0
        f64 c
    for i in range(I):
        c = a[i] - b[i]
        r += c*c
    return r

# Expand data from 2-bit to 8-bit shuffled genotype matrix
cpdef void expandShuf(
        const u8[:,::1] B, u8[:,::1] G, f64[::1] q_nrm, const u32[::1] s_ord
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t N_b = B.shape[1]
        size_t i, j, b, x, bit
        u8[4] recode = [2, 9, 1, 0]
        u8 mask = 3
        u8 byte
        u8* g
        f64* Q_cnt
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        # Thread-local buffer allocation
        Q_cnt = <f64*>calloc(N, sizeof(f64))
        if Q_cnt is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            i = 0
            g = &G[s_ord[j],0]
            for b in range(N_b):
                byte = B[j,b]
                for bit in range(4):
                    g[i] = recode[(byte >> 2*bit) & mask]
                    if g[i] != 9:
                        Q_cnt[i] += 2.0
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

# Expand data from 2-bit to 8-bit genotype matrix
cpdef void expandGeno(
        const u8[:,::1] B, u8[:,::1] G
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t N_b = B.shape[1]
        size_t i, j, b, bit
        u8[4] recode = [2, 9, 1, 0]
        u8 mask = 3
        u8 byte
        u8* g
    for j in prange(M, schedule='guided'):
        i = 0
        g = &G[j,0]
        for b in range(N_b):
            byte = B[j,b]
            for bit in range(4):
                g[i] = recode[(byte >> 2*bit) & mask]
                i = i + 1
                if i == N:
                    break

# Initialize P in supervised mode
cpdef void initP(
        u8[:,::1] G, f64[:,::1] P, const u8[::1] y
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = P.shape[1]
        size_t i, j, k
        u8* g
        f64 a
        f64* p
        f64* x
    with nogil, parallel():
        # Thread-local buffer allocation
        x = <f64*>calloc(K, sizeof(f64))
        if x is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            g = &G[j,0]
            p = &P[j,0]
            for i in range(N):
                if (g[i] != 9) and (y[i] > 0):
                    x[y[i] - 1] += 2.0
                    p[y[i] - 1] += <f64>g[i]
            for k in range(K):
                a = p[k]
                if x[k] > 0.0:
                    a = a/x[k]
                p[k] = _clamp1(a)
                x[k] = 0.0
        free(x)

# Initialize Q in supervised mode
cpdef void initQ(
        f64[:,::1] Q, const u8[::1] y
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i
    for i in prange(N, schedule='guided'):
        _begQ(&Q[i,0], y[i], K)

# Update Q in supervised mode
cpdef void superQ(
        f64[:,::1] Q, const u8[::1] y
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i
    for i in prange(N, schedule='guided'):
        if y[i] > 0:
            _setQ(&Q[i,0], y[i] - 1, K)

# Estimate minor allele frequencies
cpdef void estimateFreq(
        u8[:,::1] G, f32[::1] f
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        size_t i, j
        u8* g
        f32 c, n
    for j in prange(M, schedule='guided'):
        c = 0.0
        n = 0.0
        g = &G[j,0]
        for i in range(N):
            if g[i] != 9:
                c = c + <f32>g[i]
                n = n + 2.0
        f[j] = c/n

# Log-likelihood
cpdef f64 loglike(
        const u8[:,::1] G, f64[:,::1] P, const f64[:,::1] Q
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        size_t j
        f64 r = 0.0
        f64* h
    with nogil, parallel():
        # Thread-local buffer allocation
        h = <f64*>calloc(N, sizeof(f64))
        if h is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            _computeH(&Q[0,0], &P[j,0], h, N, K)
            r += _computeL(&G[j,0], h, N)
        free(h)
    return r

# Log-likelihood accounting for missingness
cpdef f64 loglike_missing(
        const u8[:,::1] G, f64[:,::1] P, const f64[:,::1] Q
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        size_t j
        f64 r = 0.0
        f64* h
    with nogil, parallel():
        # Thread-local buffer allocation
        h = <f64*>calloc(N, sizeof(f64))
        if h is NULL:
            abort()

        for j in prange(M, schedule='guided'):
            _computeH(&Q[0,0], &P[j,0], h, N, K)
            r += _computeM(&G[j,0], h, N)
        free(h)
    return r

# Log-likelihood in cross-validation
cpdef f64 loglike_cross(
        u8[:,::1] G, f64[:,::1] P, const f64[:,::1] Q, const u32[::1] s_ind
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = s_ind.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i, j, l
        u8* g
        f64 r = 0.0
        f64 d, h
        f64* p
    for j in prange(M, schedule='guided'):
        g = &G[j,0]
        p = &P[j,0]
        for i in range(N):
            l = s_ind[i]
            if g[l] != 9:
                h = _computeI(p, &Q[l,0], K)
                d = <f64>g[l]
                r += d*log(h) + (2.0 - d)*log(1.0 - h)
    return r

# Deviance residual in cross-validation
cpdef f64 deviance(
        u8[:,::1] G, f64[:,::1] P, const f64[:,::1] Q, const u32[::1] s_ind
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = s_ind.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i, j, l
        u8* g
        f64 r = 0.0
        f64 e = 1e-10
        f64 d, h
        f64* p
    for j in prange(M, schedule='guided'):
        g = &G[j,0]
        p = &P[j,0]
        for i in range(N):
            l = s_ind[i]
            if g[l] != 9:
                h = 2.0*_computeI(p, &Q[l,0], K)
                d = <f64>g[l]
                r += d*log(_clamp2(d/h)) + (2.0 - d)*log(_clamp2((2.0 - d)/(2.0 - h)))
    return r

# Reorder ancestral allele frequencies
cpdef void reorderP(
        f64[:,::1] P, f64[:,::1] P_new, const u32[::1] s_ord
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = P.shape[0]
        Py_ssize_t K = P.shape[1]
        size_t j, k
        f64* p_j
        f64* p_n
    for j in prange(M, schedule='guided'):
        p_j = &P[s_ord[j],0]
        p_n = &P_new[j,0]
        for k in range(K):
            p_n[k] = p_j[k]

# Reorder ancestral allele frequencies
cpdef void shuffleP(
        f64[:,::1] P, f64[:,::1] P_new, const u32[::1] s_ord
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = P.shape[0]
        Py_ssize_t K = P.shape[1]
        size_t j, k
        f64* p_j
        f64* p_n
    for j in prange(M, schedule='guided'):
        p_j = &P[j,0]
        p_n = &P_new[s_ord[j],0]
        for k in range(K):
            p_n[k] = p_j[k]

# Root-mean-square error
cpdef f64 rmse(
        const f64[:,::1] Q, const f64[:,::1] Q_pre
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        f64 r
    r = _computeR(&Q[0,0], &Q_pre[0,0], N*K)
    return sqrt(r/((<f64>N)*(<f64>K)))

# Sum-of-squares used in evaluation 
cpdef f64 sumSquare(
        u8[:,::1] G, f64[:,::1] P, const f64[:,::1] Q
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        size_t i, j
        u8* g
        f64 r = 0.0
        f64 d, h
        f64* p
    for j in prange(M, schedule='guided'):
        p = &P[j,0]
        g = &G[j,0]
        for i in range(N):
            if g[i] != 9:
                h = 2.0*_computeI(p, &Q[i,0], K)
                d = <f64>g[i]
                r += (d - h)*(d - h)
    return r

# Kullback-Leibler divergence with average for Jensen-Shannon
cpdef f64 divKL(
        f64[:,::1] A, f64[:,::1] B
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = A.shape[0]
        Py_ssize_t K = A.shape[1]
        size_t i, k
        f64 eps = 1e-10
        f64 d = 0.0
        f64 c
        f64* a
        f64* b
    for i in prange(N, schedule='guided'):
        a = &A[i,0]
        b = &B[i,0]
        for k in range(K):
            c = (a[k] + b[k])*0.5
            d += a[k]*log(a[k]/c + eps)
    return d/<f64>N
