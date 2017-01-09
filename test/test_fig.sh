#!/bin/bash

# usage: call from same directory
# will output w_real.npy, w_ref.npy and x.npy in src/results
set -e
set -u
cd "${0%/*}" # change to script location
cd ../src
mkdir -p res
echo "Generation data..."
# add/remove argument 'full' to use/not use optimization in reference
python gen_fig3.py full > ../ref/data.r
cd ../ref
start=`date +%s`
echo "Running reference GFA..."
Rscript run_convert.r > ../src/fig3_data.py
end=`date +%s`
runtime=$((end-start))
echo "(Took ${runtime} seconds)"
cd ../src
echo "Converting reference output..."
python fig3_data.py
start=`date +%s`
echo "Running our GFA..."
# change argument to use more/less iterations
python infer.py 1
end=`date +%s`
runtime=$((end-start))
echo "(Took ${runtime} seconds)"
echo "Plotting results..."
python plot_res.py

# cleanup
cd ..
rm src/fig3_data.py
rm ref/data.r
