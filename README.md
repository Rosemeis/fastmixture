# fastmixture (v1.3.0)
`fastmixture` is a new software for estimating ancestry proportions in unrelated individuals. It is significantly faster than previous model-based software while providing accurate and robust ancestry estimates.


## Table of Contents
- [Citation](#citation)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Authors and Acknowledgements](#authors-and-acknowledgements)

## Citation
Please cite our paper in [*Peer Community Journal*](https://peercommunityjournal.org/articles/10.24072/pcjournal.503/).\
Preprint also available on [BioRxiv](https://doi.org/10.1101/2024.07.08.602454).

## Installation 
To run the `fastmixture` software, you have a few options depending on your environment and preference:

1. Installing fastmixture via PyPI or Source Code

   ```bash
   # Option 1: Build and install via PyPI
   pip install fastmixture

   # Option 2: Download source and install via pip
   git clone https://github.com/Rosemeis/fastmixture.git
   cd fastmixture
   pip install .
   
   # Option 3: Download source and install in a new Conda environment
   git clone https://github.com/Rosemeis/fastmixture.git
   conda env create -f fastmixture/environment.yml
   conda activate fastmixture
   ```
   
   You can now run the program with the `fastmixture` command. For more details on running it, see the [Usage section](#usage). 


2. Using the fastmixture docker image with Docker or Apptainer
   
   If you prefer or need to use a containerized setup (especially useful in HPC environments), a pre-built fastmixture container image is available on [Docker Hub](https://hub.docker.com/r/albarema/fastmixture).

   A. Using Docker 
   1. Pull the image from  Docker Hub

   ```bash
   # Docker command
   docker pull albarema/fastmixture
   ```

   2. Run the  `fastmixture` container

   ```bash
   # Mount the directory containing the PLINK files using --volume flag (e.g. `pwd`/project-data/) 
   # Indicate the cpus available for the container to run
   # e.g. data prefix is 'toy.data' and output prefix is 'toy.fast'
   docker run --cpus=8 -v `pwd`/project-data/:/data/ albarema/fastmixture fastmixture --bfile data/toy.data --K 3 --out data/toy.fast --threads 8
   ```

   B. Using Apptainer (formerly Singularity)

   For Apptainer/Singularity users, please take a look at your HPC system's documentation for guidance. Apptainer will create the .sif image in your current working directory (pwd) by default. You will later use this image to run the software. If needed, specify a different directory and filename to store the image. Remember to bind the directories where the data is stored (`--bind`). 

   1. Pull `fastmixture` container image into a .sif file that Apptainer can use

   ```bash
   # Singularity/Apptainer
   apptainer pull <fastmixture.sif> docker://albarema/fastmixture
   ```
   2. Run  `fastmixture` container
   
   ```bash
   # Singularity/Apptainer
   apptainer run <fastmixture.sif> fastmixture --bfile data/toy.data --K 3 --out data/toy.fast --threads 8
   ```

If you run into issues with your installation on a HPC system, it could be due to a mismatch of CPU architectures between login and compute nodes (illegal instruction). You can try and remove every instance of the `march=native` compiler flag in the [setup.py](./setup.py) file which optimizes `fastmixture` to your specific hardware setup. Another alternative is to use the [uv package manager](https://docs.astral.sh/uv/), where you can run `fastmixture` in a temporary and isolated environment by simply adding `uvx` in front of the `fastmixture` command.

```bash
# uv tool run example
uvx fastmixture --bfile data/toy.data --K 3 --out data/toy.fast --threads 8
```

## Usage
`fastmixture` requires input data in binary [PLINK](https://www.cog-genomics.org/plink/1.9/input#bed) format. 
- Choose the value of `K` that best fits your data. We recommend performing principal component analysis (PCA) first as an exploratory analysis before running `fastmixture` or to use the cross-validation error defined in `ADMIXTURE` (see below).
- Use multiple seeds for your analysis to ensure robust and reliable results (e.g. ≥ 3).

```bash
# Using binary PLINK files for K=3
fastmixture --bfile data --K 3 --threads 32 --seed 1 --out test

# Outputs Q and P files (test.K3.s1.Q and test.K3.s1.P)
```

### Supervised
A supervised mode is available in `fastmixture` using `--supervised`. Provide a file of population assignments for individuals as integers in a single column file. Unknown or admixed individuals must be given a value of '0'.

```bash
# Using binary PLINK files for K=3
fastmixture --bfile data --K 3 --threads 32 --seed 1 --out super.test --supervised data.labels

# Outputs Q and P files (super.test.K3.s1.Q and super.test.K3.s1.P)
```

### Projection
A projection mode is available in `fastmixture` using `--projection`. Provide a file of pre-computed ancestral allele frequencies (P-file) from a previous `fastmixture` run based on a reference dataset. Only ancestry proportions (Q-file) are estimated in a new dataset. SNPs must be strictly overlapping between the datasets.

```bash
# fastmixture in reference dataset
fastmixture --bfile ref --K 3 --threads 32 --seed 1 --out ref.test

# fastmixture using projection mode in new dataset
fastmixture --bfile new --K 3 --threads 32 --seed 1 --out new.test --projection ref.test.K3.s1.P

# Outputs Q file (new.test.K3.s1.Q)
```

### Cross-validation (model selection)
Perform cross-validation for model selection of `K`. The cross-validation mode only works with the standard unsupervised mode.

```bash
# Perform ancestry estimation for K=3 and estimate the cross-validation error using 5 folds
fastmixture --bfile data --K 3 --threads 32 --seed 1 --out test --cv 5
```


### Extra options
* `--iter`, specify maximum number of iterations for EM algorithm (1000)
* `--tole`, specify tolerance for convergence in EM algorithm (1e-9)
* `--batches`, specify number of initial mini-batches (32)
* `--cv-tole`, specify tolerance for convergence in cross-validation (1e-7)
* `--check`, specify number of iterations performed before convergence check (5)
* `--subsample`, specify fraction of SNPs to subsample in SVD/ALS initialization (0.7)
* `--min-subsample`, minimum number of SNPs to subsample (50000)
* `--max-subsample`, maximum number of SNPs to subsample (500000)
* `--power`, specify number of power iterations in SVD (11)
* `--chunk`, number of SNPs to process at a time in randomized SVD (8192)
* `--als-iter`, specify maximum number of iterations in ALS procedure (1000)
* `--als-tole`, specify tolerance for convergence in ALS procedure (1e-4)
* `--no-freqs`, do not save ancestral allele frequencies (P-file)
* `--random-init`, random parameter initialization instead of SVD

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](./LICENSE) file for details

## Authors and Acknowledgements
- Jonas Meisner, Novo Nordisk Foundation Center for Basic Metabolic Research, University of Copenhagen 
- Cindy Santander, Computational and RNA Biology, University of Copenhagen
- Alba Refoyo Martinez, Center for Health Data Science, University of Copenhagen 
