add_executable(benchmark main.cpp func_registry.cpp)
target_compile_features(benchmark PRIVATE cxx_std_11)
target_link_libraries(benchmark PRIVATE tsne_library)
