import numpy as np
from math import ceil
from fastmixture import shared
from fastmixture import svd

##### fastmixture functions #####
### Read PLINK files
def readPlink(bfile):
	# Find length of fam-file
	N = 0
	with open(f"{bfile}.fam", "r") as fam:
		for _ in fam:
			N += 1
	N_bytes = ceil(N/4) # Length of bytes to describe N individuals

	# Read .bed file
	with open(f"{bfile}.bed", "rb") as bed:
		B = np.fromfile(bed, dtype=np.uint8, offset=3)
	assert (B.shape[0] % N_bytes) == 0, "bim file doesn't match!"
	M = B.shape[0]//N_bytes
	B.shape = (M, N_bytes)

	# Read in full genotypes into 8-bit array
	q_nrm = np.zeros(N)
	G = np.zeros((M, N), dtype=np.uint8)
	shared.expandGeno(B, G, q_nrm)
	del B
	return G, q_nrm, M, N

### SVD through eigendecomposition
def eigSVD(C):
	D, V = np.linalg.eigh(np.dot(C.T, C))
	S = np.sqrt(D)
	U = np.dot(C, V*(1.0/S))
	return np.ascontiguousarray(U[:,::-1]), np.ascontiguousarray(S[::-1]), \
		np.ascontiguousarray(V[:,::-1])

### Randomized SVD with dynamic shifts
def randomizedSVD(G, f, K, chunk, power, rng):
	M, N = G.shape
	W = ceil(M/chunk)
	a = 0.0
	L = max(K + 10, 20)
	H = np.zeros((N, L), dtype=np.float32)
	X = np.zeros((chunk, N), dtype=np.float32)
	A = rng.standard_normal(size=(M, L), dtype=np.float32)

	# Prime iteration
	for w in np.arange(W):
		M_w = w*chunk
		if w == (W-1): # Last chunk
			X = np.zeros((M - M_w, N), dtype=np.float32)
		svd.plinkChunk(G, X, f, M_w)
		H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, _, _ = eigSVD(H)
	H.fill(0.0)

	# Power iterations
	for _ in np.arange(power):
		X = np.zeros((chunk, N), dtype=np.float32)
		for w in np.arange(W):
			M_w = w*chunk
			if w == (W-1): # Last chunk
				X = np.zeros((M - M_w, N), dtype=np.float32)
			svd.plinkChunk(G, X, f, M_w)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
		H -= a*Q
		Q, S, _ = eigSVD(H)
		H.fill(0.0)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)

	# Extract singular vectors
	X = np.zeros((chunk, N), dtype=np.float32)
	for w in np.arange(W):
		M_w = w*chunk
		if w == (W-1): # Last chunk
			X = np.zeros((M - M_w, N), dtype=np.float32)
		svd.plinkChunk(G, X, f, M_w)
		A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
	U, S, V = eigSVD(A)
	U = np.ascontiguousarray(U[:,:K]*S[:K])
	V = np.ascontiguousarray(np.dot(Q, V)[:,:K])
	return U, V

### Alternating least square (ALS) for initializing Q and F
def extractFactor(U, V, f, K, iterations, tole, rng):
	M = U.shape[0]
	P = rng.random(size=(M, K), dtype=np.float32).clip(min=1e-5, max=1-(1e-5))
	I = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
	Q = 0.5*np.dot(V, np.dot(U.T, I)) + np.sum(I*f.reshape(-1,1), axis=0)
	svd.map2domain(Q)
	Q0 = np.zeros_like(Q)

	# Perform ALS iterations
	for _ in range(iterations):
		memoryview(Q0.ravel())[:] = memoryview(Q.ravel())

		# Update P
		I = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
		P = 0.5*np.dot(U, np.dot(V.T, I)) + np.outer(f, np.sum(I, axis=0))
		P.clip(min=1e-5, max=1-(1e-5), out=P)

		# Update Q
		I = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
		Q = 0.5*np.dot(V, np.dot(U.T, I)) + np.sum(I*f.reshape(-1,1), axis=0)
		svd.map2domain(Q)

		# Check convergence
		if svd.rmse(Q, Q0) < tole:
			break
	return P.astype(float), Q.astype(float)
