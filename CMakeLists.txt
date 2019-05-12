cmake_minimum_required(VERSION 3.14)
project(anysim)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wsign-conversion -O3")

set(SOURCES
        src/main.cpp
        src/io/vtk.h
        src/core/common/sources.h
        src/core/gpu/fdtd_gpu_interface.h
        src/core/gpu/fdtd_gpu_interface.cpp
        src/core/common/common_defs.h
        src/core/cpu/sources_holder.h)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

find_package(CUDA)
if(CUDA_FOUND)
    set(GPU_SOURCES
            src/core/gpu/coloring.cuh
            src/core/gpu/coloring.cu
            src/core/gpu/fdtd.cu
            src/core/gpu/fdtd.cuh)
    add_compile_definitions(GPU_BUILD)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo")
    cuda_add_library(${CMAKE_PROJECT_NAME}_gpu ${GPU_SOURCES})
    target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_NAME}_gpu)
endif()

find_package(Qt5Widgets)
if(Qt5Widgets_FOUND)
    add_compile_definitions(GUI_BUILD)
    include_directories(src/gui/include)
    add_subdirectory(src/gui)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_NAME}_gui)
endif()

