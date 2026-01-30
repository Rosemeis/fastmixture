"""
fastmixture.
Projection mode.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from fastmixture import em
from fastmixture import shared
from time import time


##### fastmixture EM functions in projection mode #####
### Update functions
# Single updates
def steps(G, P, Q, Q_tmp, q_nrm):
    em.stepQ(G, P, Q, Q_tmp)
    em.updateQ(Q, Q_tmp, q_nrm)


# Full QN update
def quasi(G, P, Q0, Q_tmp, Q1, Q2, q_nrm):
    # 1st EM step
    em.stepQ(G, P, Q0, Q_tmp)
    em.accelQ(Q0, Q1, Q_tmp, q_nrm)

    # 2nd EM step
    em.stepQ(G, P, Q1, Q_tmp)
    em.accelQ(Q1, Q2, Q_tmp, q_nrm)

    # Acceleration update
    em.jumpQ(Q0, Q1, Q2)


# Mini-batch QN update
def batQuasi(G, P, Q0, Q_tmp, Q1, Q2, q_bat, s_bat):
    # 1st EM step
    em.stepBatchQ(G, P, Q0, Q_tmp, q_bat, s_bat)
    em.batchQ(Q0, Q1, Q_tmp, q_bat)

    # 2nd EM step
    em.stepBatchQ(G, P, Q1, Q_tmp, q_bat, s_bat)
    em.batchQ(Q1, Q2, Q_tmp, q_bat)

    # Batch acceleration update
    em.jumpQ(Q0, Q1, Q2)


### fastmixture run
def fastRun(G, P, Q, q_nrm, rng, run):
    # Extract run options
    iter = run["iter"]
    tole = run["tole"]
    check = run["check"]
    batches = run["batches"]

    # Set up parameters
    M, N = G.shape
    L_nrm = np.sum(q_nrm) / 2.0
    loglike = (
        shared.loglike_missing if np.any(q_nrm < 2.0 * float(M)) else shared.loglike
    )

    # Set up containers for EM algorithm
    Q1 = np.zeros_like(Q)
    Q2 = np.zeros_like(Q)
    Q_old = np.copy(Q)
    Q_tmp = np.zeros_like(Q)
    q_bat = np.zeros(N)
    s_var = np.arange(M, dtype=np.uint32)
    M_bat = M // batches

    # Estimate initial log-likelihood
    L_old = loglike(G, P, Q)
    print(f"Initial log-like: {L_old:.1f}")
    L_bat = L_pre = L_saf = L_old

    # Parameters for stochastic EM
    safety = False
    converged = False

    # Accelerated priming iteration
    ts = time()
    steps(G, P, Q, Q_tmp, q_nrm)
    quasi(G, P, Q, Q_tmp, Q1, Q2, q_nrm)
    steps(G, P, Q, Q_tmp, q_nrm)
    print(f"Performed priming iteration.\t({time() - ts:.1f}s)\n", flush=True)

    # fastmixture algorithm
    ts = time()
    print("Estimating Q and P using mini-batch EM.")
    print(f"Using {batches} mini-batches.")
    for it in np.arange(iter):
        if batches > 1:  # Quasi-Newton mini-batch updates
            rng.shuffle(s_var)  # Shuffle SNP order
            for b in np.arange(batches):
                s_beg = b * M_bat
                s_end = M if b == (batches - 1) else (b + 1) * M_bat
                s_bat = s_var[s_beg:s_end]
                batQuasi(G, P, Q, Q_tmp, Q1, Q2, q_bat, s_bat)

            # Full updates
            if safety:  # Safety updates
                steps(G, P, Q, Q_tmp, q_nrm)
                quasi(G, P, Q, Q_tmp, Q1, Q2, q_nrm)
                steps(G, P, Q, Q_tmp, q_nrm)
            else:  # Standard updates
                quasi(G, P, Q, Q_tmp, Q1, Q2, q_nrm)
        else:  # Full updates
            if safety:  # Safety updates with log-likelihood
                quasi(G, P, Q, Q_tmp, Q1, Q2, q_nrm)
                steps(G, P, Q, Q_tmp, q_nrm)
                L_cur = loglike(G, P, Q)
                if L_cur > L_saf:
                    L_saf = L_cur
                else:  # Break and exit with best estimates
                    memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
                    converged = True
                    L_cur = L_old
                    print("No improvement. Returning with best estimate!")
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
                L_cur = loglike(G, P, Q)
                print(
                    f"({it + 1})\tLog-like: {L_cur:.1f}\t({time() - ts:.1f}s)",
                    flush=True,
                )
                if (L_cur < L_pre) and not safety:  # Check for unstable update
                    print("Turning on safety updates.")
                    memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
                    L_cur = L_bat = L_old
                    safety = True
                else:  # Check for halving
                    if (L_cur / L_nrm) < ((L_bat / L_nrm) + tole):
                        batches = batches // 2  # Halve number of batches
                        if batches > 1:
                            print(f"Halving mini-batches to {batches}.")
                            M_bat = M // batches
                            L_bat = float("-inf")
                            L_pre = L_cur
                        else:  # Turn off mini-batch acceleration
                            print("Running standard updates.")
                            L_saf = L_cur
                        if not safety:
                            quasi(G, P, Q, Q_tmp, Q1, Q2, q_nrm)
                    else:
                        L_bat = L_cur
                        if L_cur > L_old:  # Update best estimates
                            memoryview(Q_old.ravel())[:] = memoryview(Q.ravel())
                            L_old = L_cur
            else:
                if not safety:  # Estimate log-like
                    L_cur = loglike(G, P, Q)
                print(
                    f"({it + 1})\tLog-like: {L_cur:.1f}\t({time() - ts:.1f}s)",
                    flush=True,
                )
                if (L_cur < L_pre) and not safety:  # Check for unstable update
                    print("Turning on safety updates.")
                    memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
                    L_cur = L_old
                    safety = True
                else:  # Check for convergence
                    if (L_cur / L_nrm) < ((L_pre / L_nrm) + tole):
                        if L_cur < L_old:  # Use best estimates
                            memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
                            L_cur = L_old
                        converged = True
                        print("Converged!\n")
                        break
                    else:
                        L_pre = L_cur
                        if L_cur > L_old:  # Update best estimates
                            memoryview(Q_old.ravel())[:] = memoryview(Q.ravel())
                            L_old = L_cur
            ts = time()
    if not converged:
        print("Failed to converge!\n")
    print(f"Final log-likelihood: {L_cur:.1f}")
    res = {"like": L_cur, "iter": it, "conv": converged}
    return res


##### Main exception #####
assert __name__ != "__main__", "Please use the 'fastmixture' command!"
