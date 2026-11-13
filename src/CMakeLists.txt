cmake_minimum_required(VERSION 3.24)
project(graphcut_search LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(GC_DEFS
    PHMS_V3_BF_ENTRIES=399887
    PHMS_V3_PARENTS_PER_ROUND=2
    PHMS_QG_DETERMINISTIC_FILTER=1
    PHMS_PACKED_BLOOM_STORAGE
    PHMS_PACKED_BLOOM_READ_PHASE)

set(GC_HOST_FLAGS -O3 $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC,-fopenmp,-mavx2,-mfma>)
set(GC_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_library(phms_scaling SHARED phms_search.cu)
target_include_directories(phms_scaling PUBLIC ${GC_INCLUDES})
target_compile_definitions(phms_scaling PRIVATE ${GC_DEFS} PHMS_GC_GROUPS=8)
target_compile_options(phms_scaling PRIVATE ${GC_HOST_FLAGS})
target_link_libraries(phms_scaling PRIVATE CUDA::cublas OpenMP::OpenMP_CXX)

foreach(G 2 4 6 8)
  add_executable(session_${G} session.cu)
  target_include_directories(session_${G} PRIVATE ${GC_INCLUDES})
  target_compile_definitions(session_${G} PRIVATE ${GC_DEFS} PHMS_GC_GROUPS=${G})
  target_compile_options(session_${G} PRIVATE ${GC_HOST_FLAGS})
  target_link_libraries(session_${G} PRIVATE phms_scaling CUDA::cublas OpenMP::OpenMP_CXX)
  set_target_properties(session_${G} PROPERTIES
      BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
endforeach()
