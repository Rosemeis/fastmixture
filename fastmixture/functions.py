import numpy as np
import subprocess
from math import ceil
from fastmixture import em
from fastmixture import em_batch
from fastmixture import svd

##### fastmixture functions #####
### PLINK info
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, _ = process.communicate()
	return int(result.split()[0])

### Randomized SVD (PCAone Halko)
def randomizedSVD(G, f, K, batch, power, seed, threads):
	M = G.shape[0]
	N = G.shape[1]
	W = ceil(M/batch)
	L = K + 20
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
	return U, V

### Alternating least square (ALS) for initializing Q and F
def extractFactor(U, V, f, K, iterations, tole, seed):
	rng = np.random.default_rng(seed)
	M = U.shape[0]
	N = V.shape[0]
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

### SQUAREM
# Full update
def squarem(G, P, Q, P0, Q0, Q_new, dP1, dP2, dP3, dQ1, dQ2, dQ3, threads):
	np.copyto(P0, P, casting="no")
	np.copyto(Q0, Q, casting="no")

	# 1st EM step
	em.accelP(G, P, Q, Q_new, dP1, threads)
	em.accelQ(Q, Q_new, dQ1, P.shape[0], threads)

	# 2nd EM step
	em.accelP(G, P, Q, Q_new, dP2, threads)
	em.accelQ(Q, Q_new, dQ2, P.shape[0], threads)

	# Acceleation update
	em.alphaP(P, P0, dP1, dP2, dP3, threads)
	em.alphaQ(Q, Q0, dQ1, dQ2, dQ3, threads)

# Mini-batch update
def squaremBatch(G, P, Q, P0, Q0, Q_new, dP1, dP2, dP3, dQ1, dQ2, dQ3, B, \
		threads):
	np.copyto(P0, P, casting="no")
	np.copyto(Q0, Q, casting="no")

	# 1st EM step
	em_batch.accelP(G, P, Q, Q_new, dP1, B, threads)
	em.accelQ(Q, Q_new, dQ1, B.shape[0], threads)

	# 2nd EM step
	em_batch.accelP(G, P, Q, Q_new, dP2, B, threads)
	em.accelQ(Q, Q_new, dQ2, B.shape[0], threads)

	# Batch acceleration update
	em_batch.alphaP(P, P0, dP1, dP2, dP3, B, threads)
	em.alphaQ(Q, Q0, dQ1, dQ2, dQ3, threads)
