# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from libc.math cimport fmaxf, fminf, sqrtf
from libc.stdint cimport uint8_t

ctypedef uint8_t u8
ctypedef float f32

cdef f32 PRO_MIN = 1e-5
cdef f32 PRO_MAX = 1.0 - (1e-5)
cdef inline f32 _clamp3(f32 a) noexcept nogil: return fmaxf(PRO_MIN, fminf(a, PRO_MAX))


##### fastmixture - ALS/SVD optimization #####
# Normalize Q
cdef inline void _nrmQ(
        f32* q, const Py_ssize_t K
    ) noexcept nogil:
    cdef:
        size_t k
        f32 sumQ = 0.0
        f32 a, b
    for k in range(K):
        a = q[k]
        b = _clamp3(a)
        sumQ += b
        q[k] = b
    for k in range(K):
        q[k] /= sumQ

# Compute the squared difference
cdef inline f32 _computeR(
        const f32* a, const f32* b, const Py_ssize_t I
    ) noexcept nogil:
    cdef:
        size_t i
        f32 r = 0.0
        f32 c
    for i in range(I):
        c = a[i] - b[i]
        r += c*c
    return r

# Load centered chunk from PLINK file for SVD
cpdef void plinkChunk(
        u8[:,::1] G, f32[:,::1] X, const f32[::1] f, const Py_ssize_t m
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = X.shape[0]
        Py_ssize_t N = X.shape[1]
        size_t i, j, l
        u8* g
        f32 d, u
        f32* x
    for j in prange(M, schedule='static'):
        l = m + j
        u = 2.0*f[l]
        g = &G[l,0]
        x = &X[j,0]
        for i in range(N):
            d = <f32>g[i] - u
            x[i] = d if g[i] != 9 else 0.0

# Root-mean square error between two Q matrices
cpdef f32 rmseQ(
        f32[:,::1] A, f32[:,::1] B
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = A.shape[0]
        Py_ssize_t K = A.shape[1]
        f32 r
    r = _computeR(&A[0,0], &B[0,0], N*K)
    return sqrtf(r/(<f32>(N)*<f32>(K)))

# Map Q parameters to domain
cpdef void projectQ(
        f32[:,::1] Q
    ) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i
    for i in prange(N, schedule='static'):
        _nrmQ(&Q[i,0], K)

# Map P parameters to domain
cpdef void projectP(
        f32[:,::1] P
    ) noexcept nogil:
    cdef:
        Py_ssize_t M = P.shape[0]
        Py_ssize_t K = P.shape[1]
        size_t j, k
        f32 a
        f32* p
    for j in prange(M, schedule='static'):
        p = &P[j,0]
        for k in range(K):
            a = p[k]
            p[k] = _clamp3(a)