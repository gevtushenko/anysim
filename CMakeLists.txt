cmake_minimum_required(VERSION 3.14)
project(FDTD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wconversion -Wsign-conversion")

add_executable(FDTD main.cpp)