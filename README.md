## How to compile and run the benchmark
Ensure CMake version >= 3.10. Run the following commands to compile:
```shell
cmake -S . -B build
cmake --build build
```

To run the benchmark:
```shell
build/bin/benchmark <path_to_X_PCA> <path_to_Y_INIT>
```

To run the benchmark for different numbers of samples
```shell
build/bin/benchmark_full <path_to_X_PCA> <path_to_Y_INIT> <log2_of_min_samples> <log2_of_max_samples>
```

To enable debug output:
```shell
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```
and run the binary from the `build-debug` folder.
