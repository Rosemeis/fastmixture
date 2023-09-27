import numpy as np
import subprocess
from math import ceil
from src import svd
from src import em
from src import em_batch

##### fastmixture functions #####
### PLINK
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, _ = process.communicate()
	return int(result.split()[0])

### Randomized SVD (PCAone Halko)
def randomizedSVD(G, f, N, K, batch, power, seed, threads, verbose):
	M = G.shape[0]
	W = ceil(M/batch)
	L = K + 16
	rng = np.random.default_rng(seed)
	O = rng.standard_normal(size=(N, L))
	A = np.zeros((M, L))
	H = np.zeros((N, L))
	for p in range(power):
		X = np.zeros((batch, N))
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for w in range(W):
			M_w = w*batch
			if w == (W-1): # Last batch
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
	if verbose:
		print("Performed Randomized SVD.")
	return U, V

### Alternating least square (ALS) for initializing Q and F
def extractFactor(U, V, f, K, iterations, tole, seed, verbose):
	rng = np.random.default_rng(seed)
	M = U.shape[0]
	N = V.shape[0]
	P = rng.random(size=(M, K)).clip(min=1e-5, max=1-(1e-5))
	I = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
	Q = 0.5*np.dot(V, np.dot(U.T, I)) + np.sum(I*f.reshape(-1,1), axis=0)
	Q0 = np.zeros_like(Q)
	svd.map2domain(Q)

	# Perform ALS iterations
	for it in range(iterations):
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
		if verbose:
			print(f"ALS ({it}): {round(svd.rmsd(Q, Q0), 8)}")
		if svd.rmsd(Q, Q0) < tole:
			break
	return P, Q

### SQUAREM
# Full update
def squarem(G, P, Q, a, sP1, sP2, sQA, sQB, DP1, DP2, DP3, DQ1, DQ2, DQ3, threads):
	# 1st EM step
	em.accelP(G, P, Q, sQA, sQB, DP1, a, threads)
	em.accelQ(Q, sQA, sQB, DQ1, a)

	# 2nd EM step
	em.accelP(G, P, Q, sQA, sQB, DP2, a, threads)
	em.accelQ(Q, sQA, sQB, DQ2, a)

	# Acceleation update
	aQ = em.alphaQ(DQ1, DQ2, DQ3)
	aP = em.alphaP(DP1, DP2, DP3, sP1, sP2, threads)
	em.accelUpdateQ(Q, DQ1, DQ3, aQ)
	em.accelUpdateP(P, DP1, DP3, aP, threads)

# Mini-batch update
def squaremBatch(G, P, Q, a, sP1, sP2, sQA, sQB, DP1, DP2, DP3, DQ1, DQ2, DQ3, B, \
		threads):
	# 1st EM step
	em_batch.accelP(G, P, Q, sQA, sQB, DP1, a, B, threads)
	em.accelQ(Q, sQA, sQB, DQ1, a)

	# 2nd EM step
	em_batch.accelP(G, P, Q, sQA, sQB, DP2, a, B, threads)
	em.accelQ(Q, sQA, sQB, DQ2, a)

	# Batch acceleration update
	aQ = em.alphaQ(DQ1, DQ2, DQ3)
	aP = em_batch.alphaP(DP1, DP2, DP3, sP1, sP2, B, threads)
	em.accelUpdateQ(Q, DQ1, DQ3, aQ)
	em_batch.accelUpdateP(P, DP1, DP3, aP, B, threads)
