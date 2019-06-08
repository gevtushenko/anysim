project(${CMAKE_PROJECT_NAME}_io)

set(IO_SOURCES
        include/io/vtk/vtk.h
        include/io/configuration_reader.h
        src/confiuration_reader.cpp)

include_directories(include)

add_library(${CMAKE_PROJECT_NAME}_io STATIC ${IO_SOURCES})
target_link_libraries(${PROJECT_NAME} ${CMAKE_PROJECT_NAME}_core)

