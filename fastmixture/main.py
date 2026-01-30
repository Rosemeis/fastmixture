"""
fastmixture.
Main caller of fastmixture.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import sys
from fastmixture import __version__
from fastmixture import fastmixture


##### fastmixture parser #####
def main():
    # Create parser
    parser = argparse.ArgumentParser(prog="fastmixture")
    parser.add_argument("--version", action="version", version=f"v{__version__}")
    parser.add_argument(
        "-b",
        "--bfile",
        metavar="PLINK",
        help="Prefix for PLINK files (.bed, .bim, .fam)",
    )
    parser.add_argument(
        "-k", "--K", metavar="INT", type=int, help="Number of ancestral components"
    )
    parser.add_argument(
        "-t",
        "--threads",
        metavar="INT",
        type=int,
        default=1,
        help="Number of threads (1)",
    )
    parser.add_argument(
        "-s", "--seed", metavar="INT", type=int, default=42, help="Set random seed (42)"
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="OUTPUT",
        default="fastmixture",
        help="Prefix output name (fastmixture)",
    )
    parser.add_argument(
        "--iter",
        metavar="INT",
        type=int,
        default=1000,
        help="Maximum number of iterations in EM (1000)",
    )
    parser.add_argument(
        "--tole",
        metavar="FLOAT",
        type=float,
        default=1e-9,
        help="Tolerance in scaled log-likelihood units (1e-9)",
    )
    parser.add_argument(
        "--batches",
        metavar="INT",
        type=int,
        default=32,
        help="Number of initial mini-batches (32)",
    )
    parser.add_argument(
        "--supervised", metavar="FILE", help="Path to population assignment file"
    )
    parser.add_argument(
        "--projection", metavar="FILE", help="Path to ancestral allele frequencies file"
    )
    parser.add_argument(
        "--cv",
        metavar="INT",
        type=int,
        help="Number of folds in cross-validation to evaluate model fit",
    )
    parser.add_argument(
        "--cv-tole",
        metavar="FLOAT",
        type=float,
        default=1e-7,
        help="Tolerance for CV in scaled log-likelihood units (1e-7)",
    )
    parser.add_argument(
        "--check",
        metavar="INT",
        type=int,
        default=5,
        help="Number of iterations between convergence checks (5)",
    )
    parser.add_argument(
        "--subsample",
        metavar="FLOAT",
        type=float,
        default=0.7,
        help="Fraction of SNPs to subsample in SVD/ALS (0.7)",
    )
    parser.add_argument(
        "--min-subsample",
        metavar="INT",
        type=int,
        default=50000,
        help="Minimum number of SNPs to subsample in SVD/ALS (50000)",
    )
    parser.add_argument(
        "--max-subsample",
        metavar="INT",
        type=int,
        default=500000,
        help="Maximum number of SNPs to subsample in SVD/ALS (500000)",
    )
    parser.add_argument(
        "--power",
        metavar="INT",
        type=int,
        default=11,
        help="Number of power iterations in randomized SVD (11)",
    )
    parser.add_argument(
        "--chunk",
        metavar="INT",
        type=int,
        default=4096,
        help="Number of SNPs in chunk operations (4096)",
    )
    parser.add_argument(
        "--als-iter",
        metavar="INT",
        type=int,
        default=1000,
        help="Maximum number of iterations in ALS (1000)",
    )
    parser.add_argument(
        "--als-tole",
        metavar="FLOAT",
        type=float,
        default=1e-4,
        help="Tolerance for RMSE of P between iterations (1e-4)",
    )
    parser.add_argument("--no-freqs", action="store_true", help="Do not save P-matrix")
    parser.add_argument(
        "--random-init", action="store_true", help="Random initialization of parameters"
    )

    # Run fastmixture
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit()
    deaf = vars(parser.parse_args([]))
    fastmixture.main(args, deaf)


##### Define main #####
if __name__ == "__main__":
    main()
