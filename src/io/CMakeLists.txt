project(${CMAKE_PROJECT_NAME}_io)

set(IO_SOURCES
        include/io/vtk/vtk.h
        include/io/con/argagg/argagg.hpp
        include/io/configuration_reader.h
        src/configuration_reader.cpp src/con_parser.cpp include/io/con/con_parser.h include/io/hdf5/hdf5_writer.h src/hdf5_writer.cpp)

include_directories(include)

add_library(${CMAKE_PROJECT_NAME}_io STATIC ${IO_SOURCES})
target_link_libraries(${PROJECT_NAME} ${CMAKE_PROJECT_NAME}_core)

find_package(HDF5)
if (HDF5_FOUND)
    add_compile_definitions(HDF5_BUILD)
    include_directories(${HDF5_INCLUDE_DIR})
    target_link_libraries(${CMAKE_PROJECT_NAME}_io hdf5 hdf5_cpp)
endif ()
