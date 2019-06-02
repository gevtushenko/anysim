cmake_minimum_required(VERSION 3.14)
project(anysim)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA QUIET)
if(CUDA_FOUND)
    add_compile_definitions(GPU_BUILD)
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message("Install CUDA library to use GPU")
endif()

include_directories(src src/core/include)
add_subdirectory(src/core)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -g")
set(SOURCES
        src/main.cpp
        src/cpp/common_funcs.h
        src/io/vtk.h)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

find_package(Qt5Widgets QUIET)
if(Qt5Widgets_FOUND)
    add_compile_definitions(GUI_BUILD)
    include_directories(src/gui/include)
    add_subdirectory(src/gui)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_NAME}_gui)
endif()

