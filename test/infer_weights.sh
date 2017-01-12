#!/bin/bash

# usage: call from same directory
# will output w_real.npy, w_ref.npy and x.npy in src/results
set -e
set -u
cd "${0%/*}" # change to script location
cd ../ref
start=`date +%s`
echo "Running reference GFA... (optimization and full rank)"
Rscript run_convert.r > ../src/fig3_data.py
end=`date +%s`
runtime=$((end-start))
echo "(Took ${runtime} seconds)"
cd ../src
echo "Converting reference output..."
python fig3_data.py
start=`date +%s`
echo "Running our GFA/FA..."
# change argument to use more/less iterations
python infer.py 1
end=`date +%s`
runtime=$((end-start))
echo "(Took ${runtime} seconds)"
