import numpy as np
import subprocess
from math import ceil
from fastmixture import em
from fastmixture import svd

##### fastmixture functions #####
### PLINK info
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, _ = process.communicate()
	return int(result.split()[0])

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
		np.copyto(Q0, Q, casting="no")

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

### Standard update
# Full EM update
def standard(G, P, Q, Q_tmp, threads):
	# EM step
	em.updateP(G, P, Q, Q_tmp, threads)
	em.updateQ(Q, Q_tmp, G.shape[0])

### Accelerated updates
# Full QN update
def quasi(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, threads):
	# 1st EM step
	em.accelP(G, P0, P1, Q0, Q_tmp, threads)
	em.accelQ(Q0, Q1, Q_tmp, G.shape[0])

	# 2nd EM step
	em.accelP(G, P1, P2, Q1, Q_tmp, threads)
	em.accelQ(Q1, Q2, Q_tmp, G.shape[0])

	# Acceleration update
	em.alphaP(P0, P1, P2, threads)
	em.alphaQ(Q0, Q1, Q2)

# Mini-batch QN update
def quasiBatch(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, s, threads):
	# 1st EM step
	em.batchP(G, P0, P1, Q0, Q_tmp, s, threads)
	em.accelQ(Q0, Q1, Q_tmp, s.shape[0])

	# 2nd EM step
	em.batchP(G, P1, P2, Q1, Q_tmp, s, threads)
	em.accelQ(Q1, Q2, Q_tmp, s.shape[0])
	
	# Batch acceleration update
	em.alphaBatchP(P0, P1, P2, s, threads)
	em.alphaQ(Q0, Q1, Q2)
