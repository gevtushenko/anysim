project(${CMAKE_PROJECT_NAME}_io)

set(IO_SOURCES
        include/io/vtk/vtk.h
        include/io/con/argagg/argagg.hpp
        include/io/configuration_reader.h
        src/configuration_reader.cpp src/con_parser.cpp include/io/con/con_parser.h)

include_directories(include)

add_library(${CMAKE_PROJECT_NAME}_io STATIC ${IO_SOURCES})
target_link_libraries(${PROJECT_NAME} ${CMAKE_PROJECT_NAME}_core)

