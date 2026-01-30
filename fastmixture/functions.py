"""
fastmixture.
Main EM algorithm.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from fastmixture import em
from fastmixture import shared
from time import time


##### fastmixture EM functions #####
### Update functions
# Single updates
def steps(G, P, Q, Q_tmp, q_nrm, y):
    em.updateP(G, P, Q, Q_tmp)
    em.updateQ(Q, Q_tmp, q_nrm)
    if y is not None:
        shared.superQ(Q, y)


# Full QN update
def quasi(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, q_nrm, y):
    # 1st EM step
    em.accelP(G, P0, P1, Q0, Q_tmp)
    em.accelQ(Q0, Q1, Q_tmp, q_nrm)
    if y is not None:
        shared.superQ(Q1, y)

    # 2nd EM step
    em.accelP(G, P1, P2, Q1, Q_tmp)
    em.accelQ(Q1, Q2, Q_tmp, q_nrm)
    if y is not None:
        shared.superQ(Q2, y)

    # Acceleration update
    em.jumpP(P0, P1, P2)
    em.jumpQ(Q0, Q1, Q2)
    if y is not None:
        shared.superQ(Q0, y)


# Mini-batch QN update
def batQuasi(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, q_var, s_var, y):
    # 1st EM step
    em.batchP(G, P0, P1, Q0, Q_tmp, q_var, s_var)
    em.batchQ(Q0, Q1, Q_tmp, q_var)
    if y is not None:
        shared.superQ(Q1, y)

    # 2nd EM step
    em.batchP(G, P1, P2, Q1, Q_tmp, q_var, s_var)
    em.batchQ(Q1, Q2, Q_tmp, q_var)
    if y is not None:
        shared.superQ(Q2, y)

    # Batch acceleration update
    em.jumpBatchP(P0, P1, P2, s_var)
    em.jumpQ(Q0, Q1, Q2)
    if y is not None:
        shared.superQ(Q0, y)


# Single safety updates
def safSteps(G, P, Q, Q_tmp, q_nrm, y):
    em.stepP(G, P, Q)
    em.stepQ(G, P, Q, Q_tmp)
    em.updateQ(Q, Q_tmp, q_nrm)
    if y is not None:
        shared.superQ(Q, y)


# Full accelerated safety update
def safQuasi(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, q_nrm, y):
    # P steps
    em.stepAccelP(G, P0, P1, Q0)  # 1st EM step
    em.stepAccelP(G, P1, P2, Q0)  # 2nd EM step
    em.jumpP(P0, P1, P2)  # Acceleration update

    # 1st Q step
    em.stepQ(G, P0, Q0, Q_tmp)
    em.accelQ(Q0, Q1, Q_tmp, q_nrm)
    if y is not None:
        shared.superQ(Q1, y)

    # 2nd Q step
    em.stepQ(G, P0, Q1, Q_tmp)
    em.accelQ(Q1, Q2, Q_tmp, q_nrm)
    if y is not None:
        shared.superQ(Q2, y)

    # Acceleration update
    em.jumpQ(Q0, Q1, Q2)
    if y is not None:
        shared.superQ(Q0, y)


### fastmixture run
def fastRun(G, P, Q, q_nrm, y, rng, run):
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
    P1 = np.zeros_like(P)
    P2 = np.zeros_like(P)
    Q1 = np.zeros_like(Q)
    Q2 = np.zeros_like(Q)
    P_old = np.copy(P)
    Q_old = np.copy(Q)
    Q_tmp = np.zeros_like(Q)
    q_var = np.zeros(N)
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
    steps(G, P, Q, Q_tmp, q_nrm, y)
    quasi(G, P, Q, Q_tmp, P1, P2, Q1, Q2, q_nrm, y)
    steps(G, P, Q, Q_tmp, q_nrm, y)
    print(f"Performed priming iteration.\t({time() - ts:.1f}s)\n", flush=True)

    # fastmixture algorithm
    ts = time()
    print("Estimating Q and P using mini-batch EM.")
    print(f"Using {batches} mini-batches.")
    for it in np.arange(iter):
        if batches > 1:  # Accelerated mini-batch updates
            rng.shuffle(s_var)  # Shuffle SNP order
            for b in np.arange(batches):
                s_beg = b * M_bat
                s_end = M if b == (batches - 1) else (b + 1) * M_bat
                s_bat = s_var[s_beg:s_end]
                batQuasi(G, P, Q, Q_tmp, P1, P2, Q1, Q2, q_var, s_bat, y)

            # Full updates
            if safety:  # Safety updates
                safSteps(G, P, Q, Q_tmp, q_nrm, y)
                safQuasi(G, P, Q, Q_tmp, P1, P2, Q1, Q2, q_nrm, y)
                safSteps(G, P, Q, Q_tmp, q_nrm, y)
            else:  # Standard updates
                quasi(G, P, Q, Q_tmp, P1, P2, Q1, Q2, q_nrm, y)
        else:  # Full updates
            if safety:  # Safety updates with log-likelihood
                safQuasi(G, P, Q, Q_tmp, P1, P2, Q1, Q2, q_nrm, y)
                safSteps(G, P, Q, Q_tmp, q_nrm, y)
                L_cur = loglike(G, P, Q)
                if L_cur > L_saf:
                    L_saf = L_cur
                else:  # Break and exit with best estimates
                    memoryview(P.ravel())[:] = memoryview(P_old.ravel())
                    memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
                    converged = True
                    L_cur = L_old
                    print("No improvement. Returning with best estimate!")
                    break

                # Update best estimates
                if L_cur > L_old:
                    memoryview(P_old.ravel())[:] = memoryview(P.ravel())
                    memoryview(Q_old.ravel())[:] = memoryview(Q.ravel())
                    L_old = L_cur
            else:
                quasi(G, P, Q, Q_tmp, P1, P2, Q1, Q2, q_nrm, y)
                steps(G, P, Q, Q_tmp, q_nrm, y)

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
                    memoryview(P.ravel())[:] = memoryview(P_old.ravel())
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
                            quasi(G, P, Q, Q_tmp, P1, P2, Q1, Q2, q_nrm, y)
                    else:
                        L_bat = L_cur
                        if L_cur > L_old:  # Update best estimates
                            memoryview(P_old.ravel())[:] = memoryview(P.ravel())
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
                    memoryview(P.ravel())[:] = memoryview(P_old.ravel())
                    memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
                    L_cur = L_old
                    safety = True
                else:  # Check for convergence
                    if (L_cur / L_nrm) < ((L_pre / L_nrm) + tole):
                        if L_cur < L_old:  # Use best estimates
                            memoryview(P.ravel())[:] = memoryview(P_old.ravel())
                            memoryview(Q.ravel())[:] = memoryview(Q_old.ravel())
                            L_cur = L_old
                        converged = True
                        print("Converged!\n")
                        break
                    else:
                        L_pre = L_cur
                        if L_cur > L_old:  # Update best estimates
                            memoryview(P_old.ravel())[:] = memoryview(P.ravel())
                            memoryview(Q_old.ravel())[:] = memoryview(Q.ravel())
                            L_old = L_cur
            ts = time()
    if not converged:
        print("Failed to converge!\n")
    print(f"Final log-likelihood: {L_cur:.1f}")
    res = {"like": L_cur, "iter": it, "conv": converged}
    return res
