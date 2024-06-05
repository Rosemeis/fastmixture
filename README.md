# fastmixture (v0.6)

`fastmixture` is a new software for inferring ancestry proportions in unrelated individuals. It is significantly faster than previous software while maintaining robustness.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
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
`fastmixture` requires input data in [PLINK binary](https://www.cog-genomics.org/plink/1.9/input#bed) format. 
- Choose the value of `k` that best fits your data. We recommend performing a principal component analysis (PCA) first as an exploratory analysis before running `fastmixture`.
- Use multiple seeds for your analysis to ensure robust and reliable results (e.g., > 5).

```bash
# Using binary PLINK files
fastmixture --bfile data --K 3 --threads 32 --seed 1 --out test

# Outputs Q and P files (test.Q and test.P)
```

## Configuration
### Number of batches

## License
This project is licensed under the GNU General Public License - see the [LICENSE](./LICENSE) file for details

## Authors and Acknowledgements
- Jonas Meisner, [Novo Nordisk Foundation Center for Protein Research](https://www.cpr.ku.dk/staff/?pure=en/persons/433753), University of Copenhagen 
- Cindy Santander, [Computational and RNA Biology]([https://www.cpr.ku.dk/staff/?pure=en/persons/433753](https://www1.bio.ku.dk/english/staff/?pure=en%2Fpersons%2Fcindy-santander(7fb91780-169e-48b7-80ed-6741e9f3af9e).html)), University of Copenhagen
- Alba Refoyo Martinez, [Center for Health Data Science](https://heads.ku.dk/team/), University of Copenhagen 
