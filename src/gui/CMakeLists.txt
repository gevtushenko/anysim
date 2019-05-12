cmake_minimum_required(VERSION 3.14)
project(${CMAKE_PROJECT_NAME}_gui)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5OpenGL REQUIRED)

include_directories(include ${Qt5Widgets_INCLUDES} ${Qt5OpenGL_INCLUDES})
add_definitions(${Qt5Widgets_DEFINITIONS} ${Qt5OpenGL_DEFINITIONS})

set(GUI_SOURCES
        src/main_window.cpp
        src/opengl_widget.cpp
        src/render_thread.cpp
        include/main_window.h
        include/opengl_widget.h
        include/render_thread.h include/gui_simulation_manager.h src/gui_simulation_manager.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")
add_library(${PROJECT_NAME} STATIC ${GUI_SOURCES})
target_link_libraries(${PROJECT_NAME} Qt5::Widgets Qt5::OpenGL)
