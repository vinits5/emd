#!/bin/bash

PY="python3"

# Build EMD in PyTorch
cd $PWD/emd_torch/
echo $PWD
sudo $PY setup.py install
cd $PWD/..

# Build EMD in Tensorflow
cd $PWD/emd_tf/pc_distance/
make -f makefile
cd $PWD/../..

# Run Test
$PY tf_vs_pytorch.py
