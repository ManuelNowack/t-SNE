add_executable(profiling main.cpp)
target_compile_features(profiling PRIVATE cxx_std_11)
target_link_libraries(profiling PRIVATE tsne_library)
