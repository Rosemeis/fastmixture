import numpy as np
from fastmixture import em
from fastmixture import shared
from math import ceil
from time import time

##### fastmixture functions in projection mode #####
### Update functions
# Full QN update
def quasi(G, P, Q0, Q_tmp, Q1, Q2, q_nrm):
	# 1st EM step
	em.stepQ(G, P, Q0, Q_tmp)
	em.accelQ(Q0, Q1, Q_tmp, q_nrm)

	# 2nd EM step
	em.stepQ(G, P, Q1, Q_tmp)
	em.accelQ(Q1, Q2, Q_tmp, q_nrm)

	# Acceleration update
	em.alphaQ(Q0, Q1, Q2)

# Mini-batch QN update
def batQuasi(G, P, Q0, Q_tmp, Q1, Q2, q_bat, s):
	# 1st EM step
	em.stepBatchQ(G, P, Q0, Q_tmp, q_bat, s)
	em.accelBatchQ(Q0, Q1, Q_tmp, q_bat)

	# 2nd EM step
	em.stepBatchQ(G, P, Q1, Q_tmp, q_bat, s)
	em.accelBatchQ(Q1, Q2, Q_tmp, q_bat)

	# Batch acceleration update
	em.alphaQ(Q0, Q1, Q2)

# Single updates
def steps(G, P, Q, Q_tmp, q_nrm):
	em.stepQ(G, P, Q, Q_tmp)
	em.updateQ(Q, Q_tmp, q_nrm)


### fastmixture run
def fastProject(G, P, Q, Q1, Q2, Q_tmp, Q_old, q_nrm, q_bat, s, iter, \
		tole, check, batches, rng):
	# Estimate initial log-likelihood
	L_old = shared.loglike(G, P, Q)
	print(f"Initial log-like: {L_old:.1f}")

	# Parameters for stochastic EM
	M = G.shape[0]
	safety = False
	converged = False
	L_bat = L_pre = L_old
	M_bat = ceil(M/batches)

	# Accelerated priming iteration
	ts = time()
	steps(G, P, Q, Q_tmp, q_nrm)
	quasi(G, P, Q, Q_tmp, Q1, Q2, q_nrm)
	steps(G, P, Q, Q_tmp, q_nrm)
	print(f"Performed priming iteration\t({time()-ts:.1f}s)\n", flush=True)

	# fastmixture algorithm
	ts = time()
	print("Estimating Q and P using mini-batch EM.")
	print(f"Using {batches} mini-batches.")
	for it in np.arange(iter):
		if batches > 1: # Quasi-Newton mini-batch updates
			rng.shuffle(s) # Shuffle SNP order
			for b in np.arange(batches):
				s_bat = s[(b*M_bat):min((b+1)*M_bat, M)]
				batQuasi(G, P, Q, Q_tmp, Q1, Q2, q_bat, s_bat)

			# Full updates
			if safety: # Safety updates
				steps(G, P, Q, Q_tmp, q_nrm)
				quasi(G, P, Q, Q_tmp, Q1, Q2, q_nrm)
				steps(G, P, Q, Q_tmp, q_nrm)
			else: # Standard updates
				quasi(G, P, Q, Q_tmp, Q1, Q2, q_nrm)
		else: # Full updates
			if safety: # Safety updates with log-likelihood
				quasi(G, P, Q, Q_tmp, Q1, Q2, q_nrm)
				steps(G, P, Q, Q_tmp, q_nrm)
				L_cur = shared.loglike(G, P, Q)
				if L_cur > L_saf:
					L_saf = L_cur
				else: # Break and exit with best estimates
					memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
					converged = True
					L_cur = L_old
					print("No improvement. Returning with best estimate!")
					print(f"Final log-likelihood: {L_cur:.1f}")
					break

				# Update best estimates
				if L_cur > L_old:
					memoryview(Q_old.ravel())[:] = memoryview(Q.ravel())
					L_old = L_cur
			else:
				quasi(G, P, Q, Q_tmp, Q1, Q2, q_nrm)
				steps(G, P, Q, Q_tmp, q_nrm)

		# Convergence or halving check
		if (it + 1) % check == 0:
			if batches > 1:
				L_cur = shared.loglike(G, P, Q)
				print(f"({it+1})\tLog-like: {L_cur:.1f}\t({time()-ts:.1f}s)", flush=True)
				if (L_cur < L_pre) and not safety: # Check for unstable update
					print("Turning on safety updates.")
					memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
					L_cur = L_bat = L_old
					safety = True
				else: # Check for halving
					if (L_cur < L_bat) or (abs(L_cur - L_bat) < tole):
						batches = batches//2 # Halve number of batches
						if batches > 1:
							print(f"Halving mini-batches to {batches}.")
							L_bat = float('-inf')
							M_bat = ceil(M/batches)
							L_pre = L_cur
						else: # Turn off mini-batch acceleration
							print("Running standard updates.")
							L_saf = L_cur
						if not safety:
							steps(G, P, Q, Q_tmp, q_nrm)
					else:
						L_bat = L_cur
						if L_cur > L_old: # Update best estimates
							memoryview(Q_old.ravel())[:] = memoryview(Q.ravel())
							L_old = L_cur
			else:
				if not safety: # Estimate log-like
					L_cur = shared.loglike(G, P, Q)
				print(f"({it+1})\tLog-like: {L_cur:.1f}\t({time()-ts:.1f}s)", flush=True)
				if (L_cur < L_pre) and not safety: # Check for unstable update
					print("Turning on safety updates.")
					memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
					L_cur = L_old
					safety = True
				else: # Check for convergence
					if abs(L_cur - L_pre) < tole:
						if L_cur < L_old: # Use best estimates
							memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
							L_cur = L_old
						converged = True
						print("Converged!")
						print(f"Final log-likelihood: {L_cur:.1f}")
						break
					else:
						L_pre = L_cur
						if L_cur > L_old: # Update best estimates
							memoryview(Q_old.ravel())[:] = memoryview(Q.ravel())
							L_old = L_cur
			ts = time()
	return L_cur, it, converged
