enable_testing()

add_executable(tsne_test tsne_test.cpp)
target_link_libraries(tsne_test 
  tsne_library
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(tsne_test 
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
