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

add_executable(perf_joint_probs main_joint_probs.cpp)
target_link_libraries(perf_joint_probs
    PRIVATE benchmark_library
)

add_executable(affinities_benchmark affinities_benchmark.cpp)
target_link_libraries(affinities_benchmark PRIVATE tsne_library)

add_executable(benchmark_grad_desc benchmark_grad_desc.cpp)
target_link_libraries(benchmark_grad_desc PRIVATE tsne_library)

add_executable(benchmark_euclidean_dist main_euclidean_dist.cpp)
target_link_libraries(benchmark_euclidean_dist
    PRIVATE benchmark_library
)

add_executable(benchmark_euclidean_dist_low main_euclidean_dist_low.cpp)
target_link_libraries(benchmark_euclidean_dist_low
    PRIVATE benchmark_library
)
