# fastmixture (v0.93.4)
`fastmixture` is a new software for estimating ancestry proportions in unrelated individuals. It is significantly faster than previous model-based software while providing accurate and robust ancestry estimates.


## Table of Contents
- [Installation](#installation)
- [Citation](#citation)
- [Usage](#usage)
- [License](#license)
- [Authors and Acknowledgements](#authors-and-acknowledgements)

## Installation 
```bash
# Build and install via PyPI
pip install fastmixture

# or download source and install via pip
git clone https://github.com/Rosemeis/fastmixture.git
cd fastmixture
pip install .

# or download source and install in new Conda environment
git clone https://github.com/Rosemeis/fastmixture.git
conda env create -f fastmixture/environment.yml
conda activate fastmixture


# You can now run the program with the `fastmixture` command
```

### Using the docker image via Docker or Apptainer
`fastmixture` container image is available at [dockerhub](https://hub.docker.com/repository/docker/albarema/fastmixture/general)

#### Pull `fastmixture` container image 

For Apptainer/Singularity users, please refer to your HPC system's documentation for guidance. By default, Apptainer will create the .sif image in your current working directory (pwd). You will later use this image to run the software. If needed, specify a different directory and filename to store the image.

```bash
# docker command-line
docker pull albarema/fastmixture:v0.93.3
# singularity/apptainer
apptainer pull <fastmixture.sif> docker://albarema/fastmixture:v0.93.3
```

#### Run `fastmixture` container
 
```bash
# mount the directory containing the PLINK files using --volume flag (e.g. `pwd`/project-data/) 
# indicate the cpus available for the container to run
# e.g. data prefix is 'toy.data' and results prefix is 'toy.fast'
docker run --cpus=8 -v `pwd`/project-data/:/data/ albarema/fastmixture:v0.93.3 fastmixture --bfile data/toy.data --K 3 --out data/toy.fast --threads 8

# singularity/apptainer 
apptainer run <fastmixture.sif> fastmixture --bfile data/toy.data --K 3 --out data/toy.fast --threads 8
```

## Citation
Please cite our [preprint on BioRxiv](https://doi.org/10.1101/2024.07.08.602454).

## Usage
`fastmixture` requires input data in binary [PLINK](https://www.cog-genomics.org/plink/1.9/input#bed) format. 
- Choose the value of `K` that best fits your data. We recommend performing principal component analysis (PCA) first as an exploratory analysis before running `fastmixture`.
- Use multiple seeds for your analysis to ensure robust and reliable results (e.g. ≥ 5).

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

# Outputs Q and P files (super.K3.s1.Q and super.K3.s1.P)
```

### Extra options
* `--iter`, specify maximum number of iterations for EM algorithm (1000)
* `--tole`, specify tolerance for convergence in EM algorithm (0.5)
* `--batches`, specify number of initial mini-batches (32)
* `--check`, specify number of iterations performed before convergence check (5)
* `--power`, specify number of power iterations in SVD (11)
* `--chunk`, number of SNPs to process at a time in randomized SVD (8192)
* `--als-iter`, specify maximum number of iterations in ALS procedure (1000)
* `--als-tole`, specify tolerance for convergence in ALS procedure (1e-4)
* `--no-freqs`, do not save ancestral allele frequencies (P-matrix)
* `--random-init`, random parameter initialization instead of SVD
* `--safety`, only perform safety updates

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](./LICENSE) file for details

## Authors and Acknowledgements
- Jonas Meisner, Novo Nordisk Foundation Center for Basic Metabolic Research, University of Copenhagen 
- Cindy Santander, Computational and RNA Biology, University of Copenhagen
- Alba Refoyo Martinez, Center for Health Data Science, University of Copenhagen 
