## Compilation
```shell
cmake -S . -B build
cmake --build build
```
and to enable debug
```shell
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```

## Benchmark

To run the benchmark
```shell
build/bin/benchmark <path_to_X_PCA> <path_to_Y_INIT> [<log2_min_samples> <log2_max_samples>] 
```
If `<log2_min_samples>` and `<log2_max_samples>` are not defined, run benchmark for full dataset only. Otherwise, run benchmarks for `2^<log2_min_samples>` up to and including `2^<log2_min_samples>` in multiplicative steps of $2$.

## Testing

Declare and add the functions you want to test in [tsne_test.cc](test/tsne_test.cc).
To run the tests
```shell
cmake --build build
cd build/test && ctest
```
View the test logs at `build/test/Testing/Temporary`, or by adding `--output-on-failure` flag.

Tip: If you have CMake >=3.20 you can specify the `--test-dir` flag without the initial `cd` command. Otherwise go back to your previous directory with `cd -`.