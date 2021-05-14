cmake_minimum_required(VERSION 3.11)

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

# googletest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# The compiled t-SNE library code.
add_subdirectory(src)

# The benchmark executable code.
add_subdirectory(benchmark)

# test subdirectory
add_subdirectory(test)
