#!/bin/bash

module load gcc/8.2.0 cmake/3.16.5
rm -r build
cmake -S . -B build -DBUILD_GMOCK=OFF
cmake --build build

source /cluster/apps/intel/parallel_studio_xe_2020_r0/vtune_profiler/vtune-vars.sh

# vtune -collect performance-snapshot -- build/bin/profiling_joint_probs mnist2500_X.txt mnist2500_Y_init.txt
vtune -collect hotspots -- build/bin/profiling_joint_probs mnist2500_X.txt mnist2500_Y_init.txt

