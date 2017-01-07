#!/bin/bash

# usage: call from same directory
# will output w_real.npy, w_ref.npy and x.npy in src/results
set -e
cd ../src
mkdir -p res
python gen_fig3.py > ../ref/data.r
cd ../ref
Rscript run_convert.r > ../src/fig3_data.py
cd ../src
python fig3_data.py

# cleanup
cd ..
rm src/fig3_data.py
rm ref/data.r
