# Minimum required version of CMake
cmake_minimum_required(VERSION 3.12)

# Project name
project(Bang)

# Enable CUDA support (required if CUDA is present)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ../../utils
)

# Set compiler options for C++ and OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")

# Set options for nvcc
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp -std=c++14 --compiler-options '-fPIC' -O3")

# Define the shared library target (xgnnd)
cuda_add_library(xgnnd SHARED xgnnd_search.cu)

# Ensure position-independent code and separable compilation
set_target_properties(xgnnd PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify the test driver executable
find_package(Threads REQUIRED)
add_executable(xgnnd_search test_xgnnd.cpp)
target_link_libraries(xgnnd_search PRIVATE xgnnd dl)

# Clean rule (optional, CMake handles this with 'make clean')
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY CLEAN_DIRECTORIES "xgnnd.so;xgnnd_search")
