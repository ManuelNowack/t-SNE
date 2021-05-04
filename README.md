## How to compile and run the benchmark
Ensure CMake version >= 3.10. Run the following commands to compile:
```shell
cmake -S . -B build
cmake --build build
```

To run the benchmark:
```shell
build/bin/benchmark <path_to_X_PCA> <path_to_Y_INIT> [<log2_min_samples> <log2_max_samples>] 
```
If `<log2_min_samples>` and `<log2_max_samples>` are not defined, run benchmark for full dataset only. Otherwise, run benchmarks for `2^<log2_min_samples>` up to and including `2^<log2_min_samples>` in multiplicative steps of $2$.

To enable debug output:
```shell
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```
and run the binary from the `build-debug` folder.
