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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -g")
set(SOURCES
        src/main.cpp
        src/cpp/common_funcs.h)

add_executable(${CMAKE_PROJECT_NAME}-con ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME}-con ${CMAKE_PROJECT_NAME}_io)
target_compile_definitions(${CMAKE_PROJECT_NAME}-con PRIVATE CON_BUILD)

# find_package(Qt5Widgets QUIET)
if(Qt5Widgets_FOUND)
    add_compile_definitions(GUI_BUILD)
    include_directories(src/gui/include)
    add_subdirectory(src/gui)
    add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})
    target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_NAME}_gui ${CMAKE_PROJECT_NAME}_io)
endif()

option(ANYSIM_TESTS "Build the tests" ON)
if(ANYSIM_TESTS)
    enable_testing()
    add_subdirectory(external/googletest)
    mark_as_advanced(
        BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols)
    set_target_properties(gtest PROPERTIES FOLDER extern)
    set_target_properties(gtest_main PROPERTIES FOLDER extern)
    set_target_properties(gmock PROPERTIES FOLDER extern)
    set_target_properties(gmock_main PROPERTIES FOLDER extern)
    macro(package_add_test TESTNAME)
        add_executable(${TESTNAME} ${ARGN})
        target_link_libraries(${TESTNAME} gtest gmock gtest_main ${CMAKE_PROJECT_NAME}_core)
        add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
        set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    endmacro()

    set(CORE_TEST_SOURCES test/configuration_test.cpp)
    package_add_test(core_tests ${CORE_TEST_SOURCES})
endif()
