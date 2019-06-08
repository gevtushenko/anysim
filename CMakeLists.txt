cmake_minimum_required(VERSION 3.10)
project(anysim)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA QUIET)
if(CUDA_FOUND)
    add_compile_definitions(GPU_BUILD)
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message("Install CUDA library to use GPU")
endif()

include_directories(src src/core/include src/io/include)
add_subdirectory(src/core)
add_subdirectory(src/io)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -O3")
set(SOURCES
        src/main.cpp
        src/cpp/common_funcs.h)

add_executable(${CMAKE_PROJECT_NAME}-con ${SOURCES})
add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_NAME}_io)
target_link_libraries(${CMAKE_PROJECT_NAME}-con ${CMAKE_PROJECT_NAME}_io)
target_compile_definitions(${CMAKE_PROJECT_NAME}-con PRIVATE CON_BUILD)

find_package(Qt5Widgets QUIET)
if(Qt5Widgets_FOUND)
    add_compile_definitions(GUI_BUILD)
    include_directories(src/gui/include)
    add_subdirectory(src/gui)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_NAME}_gui)
endif()

