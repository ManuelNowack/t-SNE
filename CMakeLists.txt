cmake_minimum_required(VERSION 3.10)

project(tSNE VERSION 0.1.0)

# Specify the C++ standard.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ensure -std=c++xx instead of -std=g++xx.
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-tree-vectorize -march=skylake -m64 -Wall")
if (CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions(-DDEBUG_BUILD)
endif()

# Nicely support folders in IDEs.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set executable output path.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# The compiled t-SNE library code.
add_subdirectory(src)

# The benchmark executable code.
add_subdirectory(benchmark)
