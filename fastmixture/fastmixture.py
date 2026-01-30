"""
fastmixture.
Ancestry estimation.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
from datetime import datetime
from time import time
from fastmixture import __version__


##### fastmixture #####
def main(args, deaf):
    print("-------------------------------------------------")
    print(f"fastmixture v{__version__}")
    print("C.G. Santander, A. Refoyo-Martinez and J. Meisner")
    print(
        f"K={args.K}, seed={args.seed}, batches={args.batches}, threads={args.threads}"
    )
    print("-------------------------------------------------\n")

    # Check input
    assert args.bfile is not None, "No input data (--bfile)!"
    assert args.K > 1, "Please select K > 1!"
    assert args.threads > 0, "Please select a valid number of threads!"
    assert args.seed >= 0, "Please select a valid seed!"
    assert args.batches > 1, "Please select a valid number of batches!"
    assert args.iter > 0, "Please select a valid number of iterations!"
    assert args.tole >= 0.0, "Please select a valid tolerance!"
    assert args.check > 0, "Please select a valid value for convergence check!"
    assert args.min_subsample > 0, "Please select a valid number of SNPs!"
    assert args.max_subsample > 0, "Please select a valid number of SNPs!"
    assert args.power > 0, "Please select a valid number of power iterations!"
    assert args.chunk > 0, "Please select a valid value for chunk size in SVD!"
    assert args.als_iter > 0, "Please select a valid number of iterations in ALS!"
    assert args.als_tole >= 0.0, "Please select a valid tolerance in ALS!"
    assert (args.subsample > 0.0) and (args.subsample <= 1.0), (
        "Please select a valid fraction!"
    )
    if args.cv is not None:
        assert args.cv > 1, (
            "Please select a valid number of folds for cross-validation!"
        )
        assert (args.supervised is None) and (args.projection is None), (
            "Only unsupervised mode works with cross-validation!"
        )
    start = time()

    # Create log-file of used arguments
    full = vars(args)
    mand = ["seed", "batches"]
    with open(f"{args.out}.K{args.K}.s{args.seed}.log", "w") as log:
        log.write(f"fastmixture v{__version__}\n")
        log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        log.write(f"Directory: {os.getcwd()}\n")
        log.write("Options:\n")
        for key in full:
            if full[key] != deaf[key]:
                log.write(f"\t--{key}\n") if (type(full[key]) is bool) else log.write(
                    f"\t--{key} {full[key]}\n"
                )
            elif key in mand:
                log.write(f"\t--{key} {full[key]}\n")
    del full, deaf, mand

    # Control threads of external numerical libraries
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_MAX_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_MAX_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

    # Load numerical libraries
    import numpy as np
    from fastmixture import utils
    from fastmixture import shared

    ### Read data
    assert os.path.isfile(f"{args.bfile}.bed"), "bed file doesn't exist!"
    assert os.path.isfile(f"{args.bfile}.bim"), "bim file doesn't exist!"
    assert os.path.isfile(f"{args.bfile}.fam"), "fam file doesn't exist!"
    print("Reading data...", end="", flush=True)
    rng = np.random.default_rng(args.seed)  # Set up random number generator
    G, q_nrm, s_ord, M, N = utils.readPlink(args.bfile, rng)
    print(f"\rLoaded {N} samples and {M} SNPs.")
    assert not np.any(q_nrm == 0), "Sample(s) with zero information!"
    if int(M / args.batches) < 10000:
        print("\nWARNING: Few SNPs per mini-batch!\n")

    # Set up parameters
    if args.supervised is not None:  # Supervised mode
        # Check input of ancestral sources
        print("Ancestry estimation in supervised mode!")
        y = np.loadtxt(args.supervised, dtype=np.uint8).reshape(-1)
        assert y.shape[0] == N, "Number of samples differ between files!"
        assert np.max(y) <= args.K, "Wrong number of ancestral sources!"
        assert np.min(y) >= 0, "Wrong format in population assignments!"
        print(f"{np.sum(y > 0)}/{N} individuals with fixed ancestry.")

        # Count ancestral sources
        z = np.unique(y[y > 0])
        z -= 1
        z = np.sort(z)

        # Initialize parameters
        P = rng.random(size=(M, args.K))
        Q = rng.random(size=(N, args.K))
        P[:, z] = 0.0
        shared.initP(G, P, y)
        shared.initQ(Q, y)
        del z
    elif args.projection is not None:  # Projection mode
        # Check input of ancestral allele frequencies
        print("Ancestry estimation in projection mode!")
        P_raw = np.loadtxt(args.projection, dtype=float).clip(
            min=1e-5, max=1.0 - (1e-5)
        )
        assert P_raw.shape[0] == M, "Number of SNPs differ between files!"
        assert P_raw.shape[1] == args.K, "Wrong number of ancestral sources!"

        # Shuffle SNPs according to genotype matrix
        P = np.zeros_like(P_raw)
        shared.shuffleP(P_raw, P, s_ord)
        del P_raw

        # Initialize Q matrix
        Q = rng.random(size=(N, args.K)).clip(min=1e-5, max=1.0 - (1e-5))
        Q /= np.sum(Q, axis=1, keepdims=True)
    else:  # Unsupervised mode
        if args.random_init:  # Random initialization
            print("Random initialization.")
            P = rng.random(size=(M, args.K)).clip(min=1e-5, max=1.0 - (1e-5))
            Q = rng.random(size=(N, args.K)).clip(min=1e-5, max=1.0 - (1e-5))
            Q /= np.sum(Q, axis=1, keepdims=True)
        else:  # SVD-based initialization
            f = np.zeros(M, dtype=np.float32)
            shared.estimateFreq(G, f)
            assert (np.min(f) > 0.0) & (np.max(f) < 1.0), (
                "Please perform MAF filtering!"
            )

            # Initialize P and Q matrices from SVD and ALS
            print("SVD/ALS initialization.", end="", flush=True)
            ts = time()
            if (args.subsample < 1.0) and (args.min_subsample < M):  # Subsampling mode
                M_sub = int(
                    max(args.min_subsample, min(M * args.subsample, args.max_subsample))
                )
                U_sub, S, V = utils.randomSVD(
                    G, f, args.K - 1, M_sub, args.chunk, args.power, rng
                )
                U_rem = utils.projectSVD(G, S, V, f, M_sub, args.chunk)
                P, Q = utils.factorSub(
                    U_sub, U_rem, S, V, f, args.als_iter, args.als_tole, rng
                )
                del U_sub, U_rem
            else:  # Standard mode
                U, S, V = utils.randomSVD(
                    G, f, args.K - 1, M, args.chunk, args.power, rng
                )
                P, Q = utils.factorALS(U, S, V, f, args.als_iter, args.als_tole, rng)
                del U
            print(f"\rSVD/ALS initialization.\t\t({time() - ts:.1f}s)")
            del S, V, f
        y = None

    # Run options dictionary
    run = {
        "iter": args.iter,
        "tole": args.tole,
        "check": args.check,
        "batches": args.batches,
    }

    # Fastmixture
    if args.projection is not None:  # Projection mode
        from fastmixture import projection

        res = projection.fastRun(G, P, Q, q_nrm, rng, run)
    else:  # Unsupervised or supervised mode
        from fastmixture import functions

        res = functions.fastRun(G, P, Q, q_nrm, y, rng, run)

        # Cross-validation mode in unsupervised mode
        if args.cv is not None:
            from fastmixture import cross

            run["tole"] = args.cv_tole
            run["cross"] = args.cv
            res_crv = cross.crossRun(G, P, Q, q_nrm, rng, run)

    # Print elapsed time for estimation
    t_tot = time() - start
    t_min = int(t_tot // 60)
    t_sec = int(t_tot - t_min * 60)
    print(f"Total elapsed time: {t_min}m{t_sec}s")

    ### Save estimates and write output to log-file
    np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.Q", Q, fmt="%.6f")
    print(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q")
    if not args.no_freqs and (args.projection is None):  # Save frequencies
        P_ord = np.zeros_like(P)
        shared.reorderP(P, P_ord, s_ord)
        np.savetxt(f"{args.out}.K{args.K}.s{args.seed}.P", P_ord, fmt="%.6f")
        print(f"Saved P matrix as {args.out}.K{args.K}.s{args.seed}.P")

    # Write to log-file
    with open(f"{args.out}.K{args.K}.s{args.seed}.log", "a") as log:
        log.write(f"\nFinal log-likelihood: {res['like']:.1f}\n")
        if args.cv is not None:
            log.write(
                f"Cross-validation error (SD): {res_crv['avg']:.5f} ({res_crv['std']:.5f})\n"
            )
        if res["conv"]:
            log.write(f"Converged in {res['iter'] + 1} iterations.\n")
        else:
            log.write(f"EM algorithm did not converge in {args.iter} iterations!\n")
        log.write(f"Total elapsed time: {t_min}m{t_sec}s\n")
        log.write(f"Saved Q matrix as {args.out}.K{args.K}.s{args.seed}.Q\n")
        if not args.no_freqs and (args.projection is None):
            log.write(f"Saved P matrix as {args.out}.K{args.K}.s{args.seed}.P\n")


##### Main exception #####
assert __name__ != "__main__", "Please use the 'fastmixture' command!"
