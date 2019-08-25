project(${CMAKE_PROJECT_NAME}_io)

set(IO_SOURCES
        include/io/vtk/vtk.h
        include/io/con/argagg/argagg.hpp
        include/io/configuration_reader.h
        src/configuration_reader.cpp
        src/con_parser.cpp
        include/io/con/con_parser.h
        include/io/hdf5/hdf5_writer.h
        src/hdf5_writer.cpp)

add_library(${CMAKE_PROJECT_NAME}_io STATIC ${IO_SOURCES})
target_link_libraries(${PROJECT_NAME} ${CMAKE_PROJECT_NAME}_core)
target_include_directories(${CMAKE_PROJECT_NAME}_io PUBLIC include)

find_package(HDF5)
if (HDF5_FOUND)
    target_compile_definitions(${CMAKE_PROJECT_NAME}_io PRIVATE HDF5_BUILD)
    target_include_directories(${CMAKE_PROJECT_NAME}_io PRIVATE ${HDF5_INCLUDE_DIR})
    target_link_libraries(${CMAKE_PROJECT_NAME}_io hdf5 hdf5_cpp)
endif ()
