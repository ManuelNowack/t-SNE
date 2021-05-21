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


add_executable(affinities_benchmark affinities_benchmark.cpp)
target_link_libraries(affinities_benchmark PRIVATE tsne_library)
