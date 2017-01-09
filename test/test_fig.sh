#!/bin/bash

# usage: call from same directory
# will output w_real.npy, w_ref.npy and x.npy in src/results
set -e
cd "${0%/*}" # change to script location
cd ../src
mkdir -p res
echo "Generation data..."
python gen_fig3.py > ../ref/data.r
cd ../ref
echo "Running reference GFA..."
Rscript run_convert.r > ../src/fig3_data.py
cd ../src
echo "Converting reference output..."
python fig3_data.py
echo "Running our GFA..."
# change argument to use more/less iterations
python infer.py 1
echo "Plotting results..."
python plot_res.py

# cleanup
cd ..
rm src/fig3_data.py
rm ref/data.r
