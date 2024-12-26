import numpy as np
from math import ceil
from fastmixture import em
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
	Q_nrm = np.zeros(N)
	G = np.zeros((M, N), dtype=np.uint8)
	shared.expandGeno(B, G, Q_nrm)
	del B
	return G, Q_nrm, M, N

### Randomized SVD (PCAone Halko)
def randomizedSVD(G, f, K, chunk, power, rng):
	M = G.shape[0]
	N = G.shape[1]
	W = ceil(M/chunk)
	L = K + 20
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
			svd.plinkChunk(G, X, f, M_w)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, O)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	B = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(B, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, B, H, O, Q, R, R1, R2, Uhat, X
	U = np.ascontiguousarray(U[:,:K]*S[:K])
	V = np.ascontiguousarray(V[:K,:].T)
	return U, V

### Alternating least square (ALS) for initializing Q and F
def extractFactor(U, V, f, K, iterations, tole, rng):
	M = U.shape[0]
	P = rng.random(size=(M, K)).clip(min=1e-5, max=1-(1e-5))
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
	return P, Q

### Accelerated updates
# Full QN update
def quasi(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, Q_nrm, y):
	# 1st EM step
	em.accelP(G, P0, P1, Q0, Q_tmp)
	em.accelQ(Q0, Q1, Q_tmp, Q_nrm)
	if y is not None:
		shared.superQ(Q1, y)

	# 2nd EM step
	em.accelP(G, P1, P2, Q1, Q_tmp)
	em.accelQ(Q1, Q2, Q_tmp, Q_nrm)
	if y is not None:
		shared.superQ(Q2, y)

	# Acceleration update
	em.alphaP(P0, P1, P2)
	em.alphaQ(Q0, Q1, Q2)
	if y is not None:
		shared.superQ(Q0, y)

# Mini-batch QN update
def quasiBatch(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, Q_bat, y, s):
	# 1st EM step
	em.accelBatchP(G, P0, P1, Q0, Q_tmp, Q_bat, s)
	em.accelBatchQ(Q0, Q1, Q_tmp, Q_bat)
	if y is not None:
		shared.superQ(Q1, y)

	# 2nd EM step
	em.accelBatchP(G, P1, P2, Q1, Q_tmp, Q_bat, s)
	em.accelBatchQ(Q1, Q2, Q_tmp, Q_bat)
	if y is not None:
		shared.superQ(Q2, y)
	
	# Batch acceleration update
	em.alphaBatchP(P0, P1, P2, s)
	em.alphaQ(Q0, Q1, Q2)
	if y is not None:
		shared.superQ(Q0, y)

### Safety updates with independent updates
# Single updates
def steps(G, P, Q, Q_tmp, Q_nrm, y):
	em.updateP(G, P, Q, Q_tmp)
	em.updateQ(Q, Q_tmp, Q_nrm)
	if y is not None:
		shared.superQ(Q, y)

# Single safety updates
def safetySteps(G, P, Q, Q_tmp, Q_nrm, y):
	em.stepP(G, P, Q)
	em.stepQ(G, P, Q, Q_tmp)
	em.updateQ(Q, Q_tmp, Q_nrm)
	if y is not None:
		shared.superQ(Q, y)

# Full accelerated safety update
def safetyQuasi(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, Q_nrm, y):
	# P steps
	em.stepAccelP(G, P0, P1, Q0)
	em.stepAccelP(G, P1, P2, Q0)
	em.alphaP(P0, P1, P2)

	# Q steps
	em.stepQ(G, P0, Q0, Q_tmp)
	em.accelQ(Q0, Q1, Q_tmp, Q_nrm)
	if y is not None:
		shared.superQ(Q1, y)
	em.stepQ(G, P0, Q1, Q_tmp)
	em.accelQ(Q1, Q2, Q_tmp, Q_nrm)
	if y is not None:
		shared.superQ(Q2, y)
	em.alphaQ(Q0, Q1, Q2)
	if y is not None:
		shared.superQ(Q0, y)

# Full accelerated safety update with bounceback
def safetyCheck(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, Q_nrm, y, L_saf):
	# P steps
	em.stepAccelP(G, P0, P1, Q0)
	em.stepAccelP(G, P1, P2, Q0)
	em.alphaP(P0, P1, P2)

	# Q steps
	em.stepQ(G, P0, Q0, Q_tmp)
	em.accelQ(Q0, Q1, Q_tmp, Q_nrm)
	if y is not None:
		shared.superQ(Q1, y)
	em.stepQ(G, P0, Q1, Q_tmp)
	em.accelQ(Q1, Q2, Q_tmp, Q_nrm)
	if y is not None:
		shared.superQ(Q2, y)
	em.alphaQ(Q0, Q1, Q2)
	if y is not None:
		shared.superQ(Q0, y)

	# Likelihood check
	L_cur = shared.loglike(G, P0, Q0)
	if L_cur < L_saf:
		memoryview(P0.ravel())[:] = memoryview(P2.ravel())
		memoryview(Q0.ravel())[:] = memoryview(Q2.ravel())
		L_cur = shared.loglike(G, P0, Q0)
	return L_cur
