#!/bin/bash

for i in `seq 2 13`; do
  valgrind --tool=callgrind --callgrind-out-file=analysis/tsne/callgrind.out.tsne.$i --simulate-cache=yes --I1=32768,8,64 --D1=32768,8,64 --LL=3145728,12,64 build/bin/profiling_tsne mnist10k_X.txt mnist10k_Y_init.txt $i $i
done
