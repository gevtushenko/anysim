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
        include/cpp/common_funcs.h
        include/core/solver/solver.h
        include/core/sm/result_extractor.h
        include/core/solver/workspace.h
        src/solver/workspace.cpp
        include/core/grid/grid.h src/grid/grid.cpp
        src/gpu/euler_2d_gpu_interface.cpp
        include/core/sm/multiprocess.h
        src/sm/multiprocess.cpp include/core/grid/geometry.h src/grid/geometry.cpp)

find_package(MPI QUIET)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -g")
add_library(${PROJECT_NAME} STATIC ${CORE_SOURCES})
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_compile_definitions(${PROJECT_NAME} PUBLIC VTUNE_BUILD)
target_link_libraries(${PROJECT_NAME} cpp_itt)

if(MPI_FOUND)
    add_compile_definitions(MPI_BUILD)
    include_directories(${MPI_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})
else()
    message("Build without MPI support")
endif()

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC PYTHON_BUILD)
    target_link_libraries(${PROJECT_NAME} pybind11::embed)
    target_include_directories(${PROJECT_NAME} PRIVATE ${pybind11_INCLUDE_DIRS})
    message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
else ()
    message("Build without Python support")
endif()

find_package(CUDA QUIET)
if(CUDA_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC GPU_BUILD)
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
else()
    message("Install CUDA library to use GPU")
endif()

if(CUDA_FOUND)
    set(GPU_SOURCES
            include/core/gpu/coloring.cuh
            include/core/gpu/reduce.cuh
            src/gpu/euler_2d.cu
            src/gpu/coloring.cu
            src/gpu/fdtd.cu
            include/core/gpu/fdtd.cuh)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo --expt-relaxed-constexpr")
    cuda_add_library(${CMAKE_PROJECT_NAME}_gpu ${GPU_SOURCES})
    target_compile_definitions(${CMAKE_PROJECT_NAME}_gpu PUBLIC GPU_BUILD)
    target_include_directories(${CMAKE_PROJECT_NAME}_gpu PRIVATE include)
    target_link_libraries(${PROJECT_NAME} ${CMAKE_PROJECT_NAME}_gpu ${CUDA_LIBRARIES})
endif()
