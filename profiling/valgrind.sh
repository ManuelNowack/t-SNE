#!/bin/bash

module load gcc/6.3.0 cmake/3.16.5 valgrind/3.13.0
rm -r build
cmake -S . -B build -DBUILD_GMOCK=OFF
cmake --build build

valgrind --tool=callgrind --simulate-cache=yes --dump-instr=yes build/bin/profiling_joint_probs mnist2500_X.txt mnist2500_Y_init.txt
