# fastmixture

## Install and build
```bash
git clone https://github.com/Rosemeis/fastmixture.git
cd fastmixture
pip3 install .

# The "fastmixture" main caller will now be available
```

### Quick usage
```bash
# Using binary PLINK files
fastmixture --bfile data --K 3 --threads 32 --seed 1 --out test

# Outputs Q and P files (test.Q and test.P)
```
