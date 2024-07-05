# fastmixture (v0.6)

`fastmixture` is a new software for estimating ancestry proportions in unrelated individuals. It is significantly faster than previous model-based software while providing accurate and robust ancestry estimates.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage) <!-- [Configuration](#configuration) -->
- [License](#license)
- [Authors and Acknowledgements](#authors-and-acknowledgements)
  
## Installation 

```bash
git clone https://github.com/Rosemeis/fastmixture.git
cd fastmixture
pip3 install .

# The "fastmixture" main caller will now be available
```

## Usage
`fastmixture` requires input data in binary [PLINK](https://www.cog-genomics.org/plink/1.9/input#bed) format. 
- Choose the value of `K` that best fits your data. We recommend performing principal component analysis (PCA) first as an exploratory analysis before running `fastmixture`.
- Use multiple seeds for your analysis to ensure robust and reliable results (e.g. ≥ 5).

```bash
# Using binary PLINK files for K=3
fastmixture --bfile data --K 3 --threads 32 --seed 1 --out test

# Outputs Q and P files (test.K3.s1.Q and test.K3.s1.P)
```

<!-- ## Configuration
### Number of batches
`fastmixture` utilizes a mini-batch scheme for accelerating its EM algorithm. 
-->
## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](./LICENSE) file for details

## Authors and Acknowledgements
- Jonas Meisner, Novo Nordisk Foundation Center for Basic Metabolic Research, University of Copenhagen 
- Cindy Santander, Computational and RNA Biology, University of Copenhagen
- Alba Refoyo Martinez, Center for Health Data Science, University of Copenhagen 
