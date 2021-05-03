add_executable(benchmark main.cpp func_registry.cpp benchmark.cpp)
target_compile_features(benchmark PRIVATE cxx_std_11)
target_link_libraries(benchmark PRIVATE tsne_library)

add_executable(benchmark_full main_full.cpp func_registry.cpp benchmark.cpp)
target_compile_features(benchmark_full PRIVATE cxx_std_11)
target_link_libraries(benchmark_full PRIVATE tsne_library)