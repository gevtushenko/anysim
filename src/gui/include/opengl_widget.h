//
// Created by egi on 5/11/19.
//

#ifndef FDTD_OPENGL_WIDGET_H
#define FDTD_OPENGL_WIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>

#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>

#include <memory>
#include <functional>

#ifdef GPU_BUILD
#include <cuda_gl_interop.h>
#endif

#include "axes_grid.h"
#include "camera.h"

class project_manager;

class opengl_widget : public QOpenGLWidget, protected QOpenGLFunctions
{
  Q_OBJECT

public:
  opengl_widget ();
  ~opengl_widget () override;

  float *get_colors (bool use_gpu);

  float * preprocess_before_colors_fill ();
  void postprocess_after_colors_fill ();

signals:
  void widget_is_ready ();

public slots:
  void update_colors (bool use_gpu);
  void update_project (project_manager &pm);

protected:
  void initializeGL () override;
  void resizeGL (int width, int height) override;
  void paintGL () override;
  void wheelEvent (QWheelEvent *event) override;
  void mousePressEvent (QMouseEvent *event) override ;
  void mouseDoubleClickEvent (QMouseEvent *event) override ;
  void mouseReleaseEvent (QMouseEvent *event) override ;
  void mouseMoveEvent (QMouseEvent *event) override ;

private:
  bool &get_button_flag (Qt::MouseButton);

private:
  bool is_initialized = false;
  bool left_button_pressed = false;
  bool right_button_pressed = false;
  bool unsupported_button_pressed = false;

  int prev_x_position = 0;
  int prev_y_position = 0;

  std::unique_ptr<QOpenGLShaderProgram> program;

  GLint attribute_coord2d, attribute_v_color;
  GLuint vbo_vertices, vbo_colors;

  long int elements_count;
  const int vertices_per_element = 4;
  const int coords_per_vertex = 2;
  const int vertex_data_per_element = vertices_per_element * coords_per_vertex;
  const int colors_per_vertex = 3;
  const int color_data_per_element = colors_per_vertex * vertices_per_element;

#ifdef GPU_BUILD
  cudaGraphicsResource_t colors_res;
#endif

  std::unique_ptr<GLfloat[]> colors;

  float x_size;
  float y_size;

  axes_grid axes;

  float *d_colors = nullptr;

  camera camera_view;
};

#endif //FDTD_OPENGL_WIDGET_H
