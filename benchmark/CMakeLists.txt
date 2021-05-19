add_library(benchmark_library
    func_registry.cpp
    benchmark.cpp
)
target_link_libraries(benchmark_library
    PUBLIC tsne_library
)

add_executable(benchmark main.cpp)
target_link_libraries(benchmark
    PRIVATE benchmark_library
)

add_executable(perf_joint_probs perf_joint_probs.cpp)
target_link_libraries(perf_joint_probs
    PRIVATE benchmark_library
)
