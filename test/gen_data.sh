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
python gen_fig3.py > ../ref/data.r
