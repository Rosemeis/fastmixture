"""
fastmixture.
Cross-validation mode.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from fastmixture import em
from fastmixture import shared
from time import time


##### fastmixture EM functions for cross-validation #####
### Update functions
# Single updates
def cvSteps(G, P, Q, Q_tmp, q_nrm, s_ind):
    em.crossP(G, P, Q, Q_tmp, s_ind)
    em.crossQ(Q, Q_tmp, q_nrm, s_ind)


# Full QN update
def cvQuasi(G, P0, Q0, Q_tmp, P1, P2, Q1, Q2, q_nrm, s_ind):
    # 1st EM step
    em.crossAccelP(G, P0, P1, Q0, Q_tmp, s_ind)
    em.crossAccelQ(Q0, Q1, Q_tmp, q_nrm, s_ind)

    # 2nd EM step
    em.crossAccelP(G, P1, P2, Q1, Q_tmp, s_ind)
    em.crossAccelQ(Q1, Q2, Q_tmp, q_nrm, s_ind)

    # Acceleration update
    em.jumpP(P0, P1, P2)
    em.jumpCrossQ(Q0, Q1, Q2, s_ind)


# Single updates for projection
def cvProjSteps(G, P, Q, Q_tmp, q_nrm, s_ind):
    em.crossStepQ(G, P, Q, Q_tmp, s_ind)
    em.crossQ(Q, Q_tmp, q_nrm, s_ind)


# Full QN update for projection
def cvProjQuasi(G, P, Q0, Q_tmp, Q1, Q2, q_nrm, s_ind):
    # 1st EM step
    em.crossStepQ(G, P, Q0, Q_tmp, s_ind)
    em.crossAccelQ(Q0, Q1, Q_tmp, q_nrm, s_ind)

    # 2nd EM step
    em.crossStepQ(G, P, Q1, Q_tmp, s_ind)
    em.crossAccelQ(Q1, Q2, Q_tmp, q_nrm, s_ind)

    # Acceleration update
    em.jumpCrossQ(Q0, Q1, Q2, s_ind)


### fastmixture run
def crossRun(G, P, Q, q_nrm, rng, run):
    # Extract run options
    iter = run["iter"]
    tole = run["tole"]
    check = run["check"]
    cross = run["cross"]

    # Set up parameters
    N = G.shape[1]
    v_crv = np.zeros(cross)

    # Set up containers for EM algorithm
    P1 = np.zeros_like(P)
    P2 = np.zeros_like(P)
    Q1 = np.zeros_like(Q)
    Q2 = np.zeros_like(Q)
    Q_tmp = np.zeros_like(Q)
    s_ind = rng.permutation(N).astype(np.uint32)
    N_crv = N // cross

    # Cross-validation
    ts = time()
    print(f"\nPerforming cross-validation using {cross} folds.")
    for c in np.arange(cross):
        print(f"\rFold {c + 1}/{cross}", end="", flush=True)
        s_beg = c * N_crv
        s_end = N if c == (cross - 1) else (c + 1) * N_crv
        s_trn = np.concatenate((s_ind[:s_beg], s_ind[s_end:]))
        s_tst = s_ind[s_beg:s_end]

        # Copy original solutions
        P_crv = np.copy(P)
        Q_crv = np.copy(Q)

        # Estimate initial log-likelihood for training
        L_nrm = np.sum(q_nrm[s_trn] / 2.0)
        L_pre = shared.loglike_cross(G, P_crv, Q_crv, s_trn)

        # fastmixture algorithm for training set
        cvSteps(G, P_crv, Q_crv, Q_tmp, q_nrm, s_trn)
        for it in np.arange(iter):
            cvQuasi(G, P_crv, Q_crv, Q_tmp, P1, P2, Q1, Q2, q_nrm, s_trn)
            cvSteps(G, P_crv, Q_crv, Q_tmp, q_nrm, s_trn)

            # Convergence check
            if (it + 1) % check == 0:
                L_cur = shared.loglike_cross(G, P_crv, Q_crv, s_trn)
                if (L_cur / L_nrm) < ((L_pre / L_nrm) + tole):
                    break
                else:
                    L_pre = L_cur

        # Estimate initial log-likelihood for testing
        L_nrm = np.sum(q_nrm[s_tst] / 2.0)
        L_pre = shared.loglike_cross(G, P_crv, Q_crv, s_tst)

        # Projection mode for test set
        cvProjSteps(G, P_crv, Q_crv, Q_tmp, q_nrm, s_tst)
        for it in np.arange(iter):
            cvProjQuasi(G, P_crv, Q_crv, Q_tmp, Q1, Q2, q_nrm, s_tst)
            cvProjSteps(G, P_crv, Q_crv, Q_tmp, q_nrm, s_tst)

            # Convergence check
            if (it + 1) % check == 0:
                L_cur = shared.loglike_cross(G, P_crv, Q_crv, s_tst)
                if (L_cur / L_nrm) < ((L_pre / L_nrm) + tole):
                    break
                else:
                    L_pre = L_cur

        # Add cross-validation prediction error
        v_crv[c] = shared.deviance(G, P_crv, Q_crv, s_tst) / L_nrm
    print(f"\rFold {c + 1}/{cross}\t({time() - ts:.1f}s)", flush=True)
    print(f"Cross-validation error (SD): {np.mean(v_crv):.5f} ({np.std(v_crv):.5f})\n")
    res = {"avg": np.mean(v_crv), "std": np.std(v_crv)}
    return res


##### Main exception #####
assert __name__ != "__main__", "Please use the 'fastmixture' command!"
