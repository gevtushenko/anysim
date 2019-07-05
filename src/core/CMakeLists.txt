project(${CMAKE_PROJECT_NAME}_core)

set(CORE_SOURCES
        include/core/gpu/euler_2d.cuh
        include/core/gpu/euler_2d_interface.h
        include/core/common/sources.h
        include/core/cpu/sources_holder.h
        include/core/gpu/fdtd_gpu_interface.h
        include/core/sm/simulation_manager.h
        include/core/cpu/euler_2d.h
        src/sm/simulation_manager.cpp
        src/gpu/fdtd_gpu_interface.cpp
        include/core/common/common_defs.h
        src/pm/project_manager.cpp
        include/core/pm/project_manager.h
        include/core/cpu/fdtd_2d.h
        include/core/cpu/thread_pool.h
        src/cpu/thread_pool.cpp
        include/core/common/curl.h
        include/core/config/configuration.h
        src/config/configuration.cpp
        include/core/solver/solver.h
        include/core/sm/result_extractor.h
        include/core/solver/workspace.h
        src/solver/workspace.cpp
        include/core/grid/grid.h src/grid.cpp src/gpu/euler_2d_gpu_interface.cpp)


find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -g")
add_library(${CMAKE_PROJECT_NAME}_core STATIC ${CORE_SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME}_core ${CMAKE_THREAD_LIBS_INIT})

if(pybind11_FOUND)
    include_directories(include ${pybind11_INCLUDE_DIRS})
    target_link_libraries(${CMAKE_PROJECT_NAME}_core pybind11::embed)
endif()

if(CUDA_FOUND)
    set(GPU_SOURCES
            include/core/gpu/coloring.cuh
            include/core/gpu/reduce.cuh
            src/gpu/euler_2d.cu
            src/gpu/coloring.cu
            src/gpu/fdtd.cu
            include/core/gpu/fdtd.cuh)
    add_compile_definitions(GPU_BUILD)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo --expt-relaxed-constexpr")
    cuda_add_library(${CMAKE_PROJECT_NAME}_gpu ${GPU_SOURCES})
    target_link_libraries(${CMAKE_PROJECT_NAME}_core ${CMAKE_PROJECT_NAME}_gpu ${CUDA_LIBRARIES})
endif()
