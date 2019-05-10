cmake_minimum_required(VERSION 3.14)
project(FDTD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wconversion -Wsign-conversion -O3")

include_directories(src)

find_package(CUDA)

set(GPU_SOURCES
        src/core/gpu/fdtd.cu
        src/core/gpu/fdtd.cuh)

add_executable(FDTD
        src/main.cpp
        src/io/vtk.h
        src/core/common/sources.h
        src/core/gpu/fdtd_gpu_interface.h
        src/core/gpu/fdtd_gpu_interface.cpp src/core/common/common_defs.h src/core/cpu/sources_holder.h)

if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo")
    add_compile_definitions(GPU_BUILD)
    cuda_add_library(FDTD_GPU ${GPU_SOURCES})
    target_link_libraries(FDTD FDTD_GPU)
endif()
