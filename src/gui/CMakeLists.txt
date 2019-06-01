cmake_minimum_required(VERSION 3.14)
project(${CMAKE_PROJECT_NAME}_gui)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Freetype REQUIRED)

include_directories(include ${Qt5Widgets_INCLUDES} ${Qt5OpenGL_INCLUDES} ${FREETYPE_INCLUDE_DIRS})
add_definitions(${Qt5Widgets_DEFINITIONS} ${Qt5OpenGL_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

set(GUI_SOURCES
        src/main_window.cpp
        src/opengl_widget.cpp
        src/render_thread.cpp
        include/main_window.h
        include/opengl_widget.h
        include/render_thread.h
        include/gui_simulation_manager.h
        src/gui_simulation_manager.cpp
        include/axes_grid.h
        src/axes_grid.cpp include/text_renderer.h src/text_renderer.cpp include/camera.h
         src/camera.cpp include/model_widget.h src/model_widget.cpp include/tree_item.h src/tree_item.cpp include/tree_model.h src/tree_model.cpp include/graphics_widget.h src/graphics_widget.cpp)
set(GUI_RESOURCES
        resources/resources.qrc)
add_library(${PROJECT_NAME} STATIC ${GUI_SOURCES} ${GUI_RESOURCES})
target_link_libraries(${PROJECT_NAME} Qt5::Widgets Qt5::OpenGL ${FREETYPE_LIBRARIES})
