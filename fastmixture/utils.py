"""
fastmixture.
Utility functions.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from math import ceil
from fastmixture import shared
from fastmixture import svd


##### fastmixture functions #####
### Read PLINK files
def readPlink(bfile, rng):
    # Find length of fam-file
    N = 0
    with open(f"{bfile}.fam", "r") as fam:
        for _ in fam:
            N += 1
    N_bytes = ceil(N / 4)  # Length of bytes to describe N individuals

    # Read .bed file
    with open(f"{bfile}.bed", "rb") as bed:
        B = np.fromfile(bed, dtype=np.uint8, offset=3)
    assert (B.shape[0] % N_bytes) == 0, "bim file doesn't match!"
    M = B.shape[0] // N_bytes
    B = B.reshape(M, N_bytes)

    # Set up arrays
    q_nrm = np.zeros(N)
    G = np.zeros((M, N), dtype=np.uint8)

    # Expand genotypes into 8-bit array
    s_ord = np.arange(M, dtype=np.uint32)
    rng.shuffle(s_ord)
    shared.expandShuf(B, G, q_nrm, s_ord)
    del B
    return G, q_nrm, s_ord, M, N


### Read PLINK files for evaluation only
def legacyPlink(bfile):
    # Find length of fam-file
    N = 0
    with open(f"{bfile}.fam", "r") as fam:
        for _ in fam:
            N += 1
    N_bytes = ceil(N / 4)  # Length of bytes to describe N individuals

    # Read .bed file
    with open(f"{bfile}.bed", "rb") as bed:
        B = np.fromfile(bed, dtype=np.uint8, offset=3)
    assert (B.shape[0] % N_bytes) == 0, "bim file doesn't match!"
    M = B.shape[0] // N_bytes
    B = B.reshape(M, N_bytes)

    # Set up array
    G = np.zeros((M, N), dtype=np.uint8)

    # Expand genotypes into 8-bit array
    shared.expandGeno(B, G)
    del B
    return G, M, N


### SVD through eigendecomposition
def eigSVD(C):
    D, V = np.linalg.eigh(np.dot(C.T, C))
    S = np.sqrt(D)
    U = np.dot(C, V * (1.0 / S))
    return (
        np.ascontiguousarray(U[:, ::-1]),
        np.ascontiguousarray(S[::-1]),
        np.ascontiguousarray(V[:, ::-1]),
    )


### Randomized SVD with dynamic shifts
def randomSVD(G, f, K, M, chunk, power, rng):
    N = G.shape[1]
    W = ceil(M / chunk)
    a = 0.0
    L = max(K + 10, 20)
    H = np.zeros((N, L), dtype=np.float32)
    X = np.zeros((chunk, N), dtype=np.float32)
    A = rng.standard_normal(size=(M, L), dtype=np.float32)

    # Prime iteration
    for w in np.arange(W):
        M_w = w * chunk
        if w == (W - 1):  # Last chunk
            X = np.zeros((M - M_w, N), dtype=np.float32)
        svd.plinkChunk(G, X, f, M_w)
        H += np.dot(X.T, A[M_w : (M_w + X.shape[0])])
    Q, _, _ = eigSVD(H)
    H.fill(0.0)

    # Power iterations
    for _ in np.arange(power):
        X = np.zeros((chunk, N), dtype=np.float32)
        for w in np.arange(W):
            M_w = w * chunk
            if w == (W - 1):  # Last chunk
                X = np.zeros((M - M_w, N), dtype=np.float32)
            svd.plinkChunk(G, X, f, M_w)
            A[M_w : (M_w + X.shape[0])] = np.dot(X, Q)
            H += np.dot(X.T, A[M_w : (M_w + X.shape[0])])
        H -= a * Q
        Q, S, _ = eigSVD(H)
        H.fill(0.0)
        if S[-1] > a:
            a = 0.5 * (S[-1] + a)

    # Extract singular vectors
    X = np.zeros((chunk, N), dtype=np.float32)
    for w in np.arange(W):
        M_w = w * chunk
        if w == (W - 1):  # Last chunk
            X = np.zeros((M - M_w, N), dtype=np.float32)
        svd.plinkChunk(G, X, f, M_w)
        A[M_w : (M_w + X.shape[0])] = np.dot(X, Q)
    U, S, V = eigSVD(A)
    U = np.ascontiguousarray(U[:, :K])
    S = np.ascontiguousarray(S[:K])
    V = np.ascontiguousarray(np.dot(Q, V)[:, :K])
    return U, S, V


### Alternating least square (ALS) for initializing Q and P
def factorALS(U, S, V, f, iter, tole, rng):
    M, K = U.shape
    Z = np.ascontiguousarray(U * S)
    P = rng.random(size=(M, K + 1), dtype=np.float32).clip(min=1e-5, max=1 - (1e-5))
    H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
    Q = 0.5 * np.dot(V, np.dot(Z.T, H)) + np.sum(H * f.reshape(-1, 1), axis=0)
    svd.projectQ(Q)
    Q0 = np.copy(Q)

    # Perform ALS iterations
    for _ in range(iter):
        # Update P
        H = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
        P = 0.5 * np.dot(Z, np.dot(V.T, H)) + np.outer(f, np.sum(H, axis=0))
        svd.projectP(P)

        # Update Q
        H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
        Q = 0.5 * np.dot(V, np.dot(Z.T, H)) + np.sum(H * f.reshape(-1, 1), axis=0)
        svd.projectQ(Q)

        # Check convergence
        if svd.rmseQ(Q, Q0) < tole:
            break
        memoryview(Q0.ravel())[:] = memoryview(Q.ravel())
    return P.astype(float), Q.astype(float)


### Projection onto PC space
def projectSVD(G, S, V, f, B, chunk):
    N = G.shape[1]
    K = V.shape[1]
    M = G.shape[0] - B
    W = ceil(M / chunk)
    Z = np.ascontiguousarray(V * (1.0 / S))
    U = np.zeros((M, K), dtype=np.float32)
    X = np.zeros((chunk, N), dtype=np.float32)

    # Loop through chunks
    for w in np.arange(W):
        M_w = w * chunk
        if w == (W - 1):  # Last chunk
            X = np.zeros((M - M_w, N), dtype=np.float32)
        svd.plinkChunk(G, X, f, B + M_w)
        U[M_w : (M_w + X.shape[0])] = np.dot(X, Z)
    return U


### Least square (ALS) for subsampled P and Q followed by standard iteration
def factorSub(U_sub, U_rem, S, V, f, iter, tole, rng):
    B, K = U_sub.shape
    u = f[:B]
    Z = np.ascontiguousarray(U_sub * S)
    P = rng.random(size=(B, K + 1), dtype=np.float32).clip(min=1e-5, max=1 - (1e-5))
    H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
    Q = 0.5 * np.dot(V, np.dot(Z.T, H)) + np.sum(H * u.reshape(-1, 1), axis=0)
    svd.projectQ(Q)
    Q0 = np.copy(Q)

    # Perform ALS iterations on subsampled SNPs
    for _ in range(iter):
        # Update P
        H = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
        P = 0.5 * np.dot(Z, np.dot(V.T, H)) + np.outer(u, np.sum(H, axis=0))
        svd.projectP(P)

        # Update Q
        H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
        Q = 0.5 * np.dot(V, np.dot(Z.T, H)) + np.sum(H * u.reshape(-1, 1), axis=0)
        svd.projectQ(Q)

        # Check convergence
        if svd.rmseQ(Q, Q0) < tole:
            break
        memoryview(Q0.ravel())[:] = memoryview(Q.ravel())
    del Q0

    # Perform extra full ALS iteration
    Z = np.ascontiguousarray(np.concatenate((U_sub, U_rem), axis=0) * S)
    H = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
    P = 0.5 * np.dot(Z, np.dot(V.T, H)) + np.outer(f, np.sum(H, axis=0))
    svd.projectP(P)
    H = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
    Q = 0.5 * np.dot(V, np.dot(Z.T, H)) + np.sum(H * f.reshape(-1, 1), axis=0)
    svd.projectQ(Q)
    return P.astype(float), Q.astype(float)


##### Main exception #####
assert __name__ != "__main__", "Please use the 'fastmixture' command!"