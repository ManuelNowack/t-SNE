add_executable(testing main.cpp base.cpp base.h tests.h)
target_compile_features(testing PRIVATE cxx_std_11)
target_link_libraries(testing PRIVATE tsne_library)