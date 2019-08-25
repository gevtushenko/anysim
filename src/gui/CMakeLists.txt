project(${CMAKE_PROJECT_NAME}_gui)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5OpenGL REQUIRED)

set(GUI_SOURCES
        src/main_window.cpp
        src/opengl_widget.cpp
        src/render_thread.cpp
        include/main_window.h
        include/opengl_widget.h
        include/render_thread.h
        include/gui_application.h
        src/gui_application.cpp
        include/axes_grid.h
        src/axes_grid.cpp
        include/camera.h
        src/camera.cpp include/model_widget.h
        src/model_widget.cpp
        include/graphics_widget.h
        src/graphics_widget.cpp
        include/settings_widget.h
        src/settings_widget.cpp
        include/settings/source_settings_widget.h
        src/settings/source_settings_widget.cpp
        include/settings/global_parameters_widget.h
        src/settings/global_parameters_widget.cpp
        include/python_syntax_highlighter.h src/python_syntax_highlighter.cpp include/section_name.h src/section_name.cpp)

set(GUI_RESOURCES resources/resources.qrc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")
add_library(${PROJECT_NAME} STATIC ${GUI_SOURCES} ${GUI_RESOURCES})
target_compile_definitions(${PROJECT_NAME} PRIVATE ${Qt5Widgets_DEFINITIONS} ${Qt5OpenGL_DEFINITIONS})
target_include_directories(${PROJECT_NAME} PUBLIC include ${Qt5Widgets_INCLUDES} ${Qt5OpenGL_INCLUDES})
target_link_libraries(${PROJECT_NAME} ${CMAKE_PROJECT_NAME}_core Qt5::Widgets Qt5::OpenGL ${CMAKE_PROJECT_NAME}_io)
