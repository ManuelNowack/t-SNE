## Compilation
CMake version >= 3.10 is required.

### CMake version >= 3.13
```shell
cmake -S . -B build
cmake --build build
```
and to enable debug
```shell
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```

### CMake version < 3.13
```shell
cmake -Bbuild -H.
cmake --build build
```
and to enable debug
```shell
cmake -Bbuild-debug -H. -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```

## Benchmark

To run the benchmark
```shell
build/bin/benchmark <path_to_X_PCA> <path_to_Y_INIT> [<log2_min_samples> <log2_max_samples>] 
```
If `<log2_min_samples>` and `<log2_max_samples>` are not defined, run benchmark for full dataset only. Otherwise, run benchmarks for `2^<log2_min_samples>` up to and including `2^<log2_min_samples>` in multiplicative steps of $2$.

## Profiling

To generate the callgrind
```shell
valgrind --tool=callgrind build-debug/bin/profiling <path_to_X_PCA> <path_to_Y_INIT>
```
and then open `callgrind.out.<pid>` in KCachegrind.

Since generating callgrinds takes a long time, they should be committed to `callgrind/callgrind.out.<tsne_func>`.