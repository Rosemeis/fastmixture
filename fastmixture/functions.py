import numpy as np
from math import ceil
from fastmixture import em
from fastmixture import shared
from fastmixture import svd

##### fastmixture functions #####
### Read PLINK files
def readPlink(bfile, threads):
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
	G = np.zeros((M, N), dtype=np.uint8)
	shared.expandGeno(B, G, threads)
	del B
	return G, M, N

### Randomized SVD (PCAone Halko)
def randomizedSVD(G, f, K, chunk, power, seed, threads):
	M = G.shape[0]
	N = G.shape[1]
	W = ceil(M/chunk)
	L = K + 20
	rng = np.random.default_rng(seed)
	O = rng.standard_normal(size=(N, L))
	A = np.zeros((M, L))
	H = np.zeros((N, L))
	for p in range(power):
		X = np.zeros((chunk, N))
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for w in range(W):
			M_w = w*chunk
			if w == (W-1): # Last chunk
				del X # Ensure no extra copy
				X = np.zeros((M - M_w, N))
			svd.plinkChunk(G, X, f, M_w, threads)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, O)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, R = np.linalg.qr(A, mode="reduced")
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, Uhat, X
	U = np.ascontiguousarray(U[:,:K]*S[:K])
	V = np.ascontiguousarray(V[:K,:].T)
	return U, V

### Alternating least square (ALS) for initializing Q and F
def extractFactor(U, V, f, K, iterations, tole, seed):
	rng = np.random.default_rng(seed)
	M = U.shape[0]
	P = rng.random(size=(M, K)).clip(min=1e-5, max=1-(1e-5))
	I = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
	Q = 0.5*np.dot(V, np.dot(U.T, I)) + np.sum(I*f.reshape(-1,1), axis=0)
	svd.map2domain(Q)
	Q0 = np.zeros_like(Q)

	# Perform ALS iterations
	for _ in range(iterations):
		shared.copyQ(Q0, Q)

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
	return P, Q

### Accelerated updates
# Full QN update
def quasi(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, y, threads):
	# 1st EM step
	em.accelP(G, P0, P1, Q0, Q_tmp, threads)
	em.accelQ(Q0, Q1, Q_tmp, G.shape[0])
	if y is not None:
		shared.superQ(Q1, y)

	# 2nd EM step
	em.accelP(G, P1, P2, Q1, Q_tmp, threads)
	em.accelQ(Q1, Q2, Q_tmp, G.shape[0])
	if y is not None:
		shared.superQ(Q2, y)

	# Acceleration update
	em.alphaP(P0, P1, P2, threads)
	em.alphaQ(Q0, Q1, Q2)
	if y is not None:
		shared.superQ(Q0, y)

# Mini-batch QN update
def quasiBatch(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, y, s, threads):
	# 1st EM step
	em.accelBatchP(G, P0, P1, Q0, Q_tmp, s, threads)
	em.accelQ(Q0, Q1, Q_tmp, s.shape[0])
	if y is not None:
		shared.superQ(Q1, y)

	# 2nd EM step
	em.accelBatchP(G, P1, P2, Q1, Q_tmp, s, threads)
	em.accelQ(Q1, Q2, Q_tmp, s.shape[0])
	if y is not None:
		shared.superQ(Q2, y)
	
	# Batch acceleration update
	em.alphaBatchP(P0, P1, P2, s, threads)
	em.alphaQ(Q0, Q1, Q2)
	if y is not None:
		shared.superQ(Q0, y)

### Safety updates with independent updates
# Single updates
def steps(G, P, Q, Q_tmp, y, threads):
	em.updateP(G, P, Q, Q_tmp, threads)
	em.updateQ(Q, Q_tmp, G.shape[0])
	if y is not None:
		shared.superQ(Q, y)

# Single safety updates
def safetySteps(G, P, Q, Q_tmp, y, threads):
	em.stepP(G, P, Q, threads)
	em.stepQ(G, P, Q, Q_tmp, threads)
	em.updateQ(Q, Q_tmp, G.shape[0])
	if y is not None:
		shared.superQ(Q, y)

# Full accelerated safety update
def safety(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, y, threads):
	# P steps
	em.stepAccelP(G, P0, P1, Q0, threads)
	em.stepAccelP(G, P1, P2, Q0, threads)
	em.alphaP(P0, P1, P2, threads)

	# Q steps
	em.stepQ(G, P0, Q0, Q_tmp, threads)
	em.accelQ(Q0, Q1, Q_tmp, G.shape[0])
	if y is not None:
		shared.superQ(Q1, y)
	em.stepQ(G, P0, Q1, Q_tmp, threads)
	em.accelQ(Q1, Q2, Q_tmp, G.shape[0])
	if y is not None:
		shared.superQ(Q2, y)
	em.alphaQ(Q0, Q1, Q2)
	if y is not None:
		shared.superQ(Q0, y)

# Full accelerated safety update with bounceback
def safetyCheck(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, y, l_vec, L_saf, threads):
	# P steps
	em.stepAccelP(G, P0, P1, Q0, threads)
	em.stepAccelP(G, P1, P2, Q0, threads)
	em.alphaP(P0, P1, P2, threads)

	# Q steps
	em.stepQ(G, P0, Q0, Q_tmp, threads)
	em.accelQ(Q0, Q1, Q_tmp, G.shape[0])
	if y is not None:
		shared.superQ(Q1, y)
	em.stepQ(G, P0, Q1, Q_tmp, threads)
	em.accelQ(Q1, Q2, Q_tmp, G.shape[0])
	if y is not None:
		shared.superQ(Q2, y)
	em.alphaQ(Q0, Q1, Q2)
	if y is not None:
		shared.superQ(Q0, y)

	# Likelihood check
	shared.loglike(G, P0, Q0, l_vec, threads)
	L_cur = np.sum(l_vec)
	if L_cur < L_saf:
		shared.copyP(P0, P2, threads)
		shared.copyQ(Q0, Q2)
		shared.loglike(G, P0, Q0, l_vec, threads)
		L_cur = np.sum(l_vec)
	return L_cur
