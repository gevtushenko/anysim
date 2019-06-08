project(${CMAKE_PROJECT_NAME}_core)

set(CORE_SOURCES
        include/core/common/sources.h
        include/core/cpu/sources_holder.h
        include/core/gpu/fdtd_gpu_interface.h
        src/gpu/fdtd_gpu_interface.cpp
        include/core/common/common_defs.h
        src/pm/project_manager.cpp
        include/core/pm/project_manager.h include/core/cpu/fdtd_2d.h include/core/cpu/thread_pool.h src/cpu/thread_pool.cpp include/core/common/curl.h)

include_directories(include)
add_library(${CMAKE_PROJECT_NAME}_core STATIC ${CORE_SOURCES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -O3")

if(CUDA_FOUND)
    set(GPU_SOURCES
            include/core/gpu/coloring.cuh
            src/gpu/coloring.cu
            src/gpu/fdtd.cu
            include/core/gpu/fdtd.cuh)
    add_compile_definitions(GPU_BUILD)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo")
    cuda_add_library(${CMAKE_PROJECT_NAME}_gpu ${GPU_SOURCES})
    target_link_libraries(${CMAKE_PROJECT_NAME}_core ${CMAKE_PROJECT_NAME}_gpu ${CUDA_LIBRARIES})
endif()
