add_executable(benchmark main.cpp func_registry.cpp benchmark.cpp)
add_executable(perf_joint_probs perf_joint_probs.cpp func_registry.cpp benchmark.cpp)
target_link_libraries(benchmark PRIVATE tsne_library)
target_link_libraries(perf_joint_probs PRIVATE tsne_library)