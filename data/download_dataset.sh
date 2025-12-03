#!/bin/bash

# Download GDB9 (~2GB)
echo "Downloading GDB9 (~2GB)..."
curl -L -O http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz

echo "Extracting..."
tar xvzf gdb9.tar.gz

echo "Cleaning up..."
rm gdb9.tar.gz

# Download NP and SA score models
echo "Downloading NP and SA score models..."
curl -L -O https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz
curl -L -O https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz

echo "Done!"