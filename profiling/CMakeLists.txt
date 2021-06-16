add_executable(profiling_joint_probs joint_probs/main.cpp)
add_executable(profiling_tsne tsne/main.cpp)

target_link_libraries(profiling_joint_probs PRIVATE tsne_library)
target_link_libraries(profiling_tsne PRIVATE tsne_library)

