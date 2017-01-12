#!/bin/bash

# usage: call from same directory
# will output w_real.npy, w_ref.npy and x.npy in src/results
set -e
set -u
cd "${0%/*}" # change to script location
cd ../src
echo "Plotting results..."
python plot_res.py
