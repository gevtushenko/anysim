//
// Created by egi on 5/11/19.
//

#include "gui/include/opengl_widget.h"
#include "core/pm/project_manager.h"
#include "cpp/common_funcs.h"
#include "core/grid/grid.h"

#include <QDirIterator>
#include <QWheelEvent>
#include <QPainter>

#include <cmath>
#include <iostream>

opengl_widget::opengl_widget ()
{
  setMinimumHeight (100);
  setMinimumWidth (100);
}

opengl_widget::~opengl_widget ()
{
  if (is_initialized)
  {
#ifdef GPU_BUILD
    cudaGraphicsUnregisterResource (colors_res);
#endif
    glDeleteBuffers (1, &vbo_vertices);
    glDeleteBuffers (1, &vbo_colors);
  }
}

float* opengl_widget::preprocess_before_colors_fill()
{
#ifdef GPU_BUILD
  size_t size = 0;
  float *colors_ptr = nullptr;
  cudaGraphicsMapResources (1, &colors_res);
  cudaGraphicsResourceGetMappedPointer ((void**) &colors_ptr, &size, colors_res);

  auto error = cudaGetLastError ();

  if (error != cudaSuccess)
    std::cout << cudaGetErrorString (error) << std::endl;

  return colors_ptr;
#else
  return nullptr;
#endif
}

void opengl_widget::postprocess_after_colors_fill()
{
#ifdef GPU_BUILD
  cudaGraphicsUnmapResources (1, &colors_res);
#endif
}

void opengl_widget::initializeGL()
{
  initializeOpenGLFunctions ();

  axes.initialize_gl (this);

  program = std::make_unique<QOpenGLShaderProgram> (this);
  program->addShaderFromSourceFile (QOpenGLShader::Vertex,   ":/shaders/map_2d.vert");
  program->addShaderFromSourceFile (QOpenGLShader::Fragment, ":/shaders/map_2d.frag");
  program->link ();
  attribute_coord2d = program->attributeLocation ("coord2d");
  attribute_v_color = program->attributeLocation ("v_color");

  glGenBuffers (1, &vbo_vertices);
  glGenBuffers (1, &vbo_colors);

  emit widget_is_ready ();
}

float *opengl_widget::get_colors (bool use_gpu)
{
  return use_gpu ? d_colors : colors.get ();
}

void opengl_widget::resizeGL(int width, int height)
{
  camera_view.resize (width, height);
  axes.resize (width, height);
}

void opengl_widget::update_colors (bool use_gpu)
{
  if (!use_gpu)
  {
    const int glfloat_size = sizeof (GLfloat);
    const long int colors_array_size = elements_count * color_data_per_element * glfloat_size;

    glBindBuffer (GL_ARRAY_BUFFER, vbo_colors);
    glBufferData (GL_ARRAY_BUFFER, colors_array_size, colors.get (), GL_DYNAMIC_DRAW);
  }

  update ();
}

void opengl_widget::update_project (project_manager &pm)
{
  const auto& solver_grid = pm.get_grid ();

  is_initialized = false;

  x_size = solver_grid.get_bounding_box_width ();
  y_size = solver_grid.get_bounding_box_height ();

  axes.prepare (0.0, x_size, 0.0, y_size);
  camera_view.update_model_matrix (x_size, y_size);

  elements_count = solver_grid.get_cells_number ();

  colors.reset ();
  colors.reset (new GLfloat[color_data_per_element * elements_count]);

  static GLfloat _colors[] =
      {
          1.0, 1.0, 0.0,
          0.0, 0.0, 1.0,
          1.0, 0.0, 0.0,
          1.0, 0.0, 1.0,
      };

  for (unsigned int i = 0; i < elements_count; i++)
  {
    const unsigned int color_offset = static_cast<unsigned int> (color_data_per_element) * i;
    std::copy_n (_colors, color_data_per_element, colors.get () + color_offset);
  }

  /// VBO Handling
  const int glfloat_size = sizeof (GLfloat);
  const long int vertices_array_size = elements_count * vertex_data_per_element * glfloat_size;
  glBindBuffer (GL_ARRAY_BUFFER, vbo_vertices);
  glBufferData (GL_ARRAY_BUFFER, vertices_array_size, solver_grid.get_vertices_data (), GL_DYNAMIC_DRAW);

  const long int colors_array_size = elements_count * color_data_per_element * glfloat_size;
  glBindBuffer (GL_ARRAY_BUFFER, vbo_colors);
  glBufferData (GL_ARRAY_BUFFER, colors_array_size, colors.get (), GL_DYNAMIC_DRAW);

#ifdef GPU_BUILD
  cudaGraphicsGLRegisterBuffer (&colors_res, vbo_colors, cudaGraphicsMapFlagsWriteDiscard);

  d_colors = preprocess_before_colors_fill ();
  postprocess_after_colors_fill ();

  auto error = cudaGetLastError ();

  if (error != cudaSuccess)
    std::cout << cudaGetErrorString (error) << std::endl;
#endif

  is_initialized = true;
}

void opengl_widget::wheelEvent(QWheelEvent *event)
{
  camera_view.zoom (event->delta ());
  update ();
}

bool &opengl_widget::get_button_flag (Qt::MouseButton button)
{
  if (button == Qt::LeftButton)
    return left_button_pressed;
  else if (button == Qt::RightButton)
    return right_button_pressed;

  return unsupported_button_pressed;
}

void opengl_widget::mousePressEvent (QMouseEvent *event)
{
  get_button_flag (event->button ()) = true;

  prev_x_position = event->x ();
  prev_y_position = event->y ();
}

void opengl_widget::mouseReleaseEvent (QMouseEvent *event)
{
  get_button_flag (event->button ()) = false;
}

void opengl_widget::mouseDoubleClickEvent (QMouseEvent *event)
{
  cpp_unreferenced (event);

  camera_view.reset ();
  update ();
}

void opengl_widget::mouseMoveEvent (QMouseEvent *event)
{
  if (left_button_pressed)
  {
    const int dx = prev_x_position - event->x ();
    const int dy = prev_y_position - event->y ();

    prev_x_position = event->x ();
    prev_y_position = event->y ();

    camera_view.move (dx, dy);
    update ();
  }
}

void opengl_widget::paintGL()
{
  if (!is_initialized)
  {
    glClear (GL_COLOR_BUFFER_BIT);
    glClearColor (1.0f, 1.0f, 1.0f, 1.0f);
    return;
  }

  QPainter painter (this);
  painter.beginNativePainting ();

  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_LINE_SMOOTH);
  glClear (GL_COLOR_BUFFER_BIT);
  glClearColor (1.0f, 1.0f, 1.0f, 1.0f);

  auto mvp = camera_view.get_mvp ();

  program->bind ();

  program->setUniformValue ("MVP", mvp);
  glEnableVertexAttribArray (static_cast<GLuint> (attribute_v_color));
  glBindBuffer (GL_ARRAY_BUFFER, vbo_colors);
  glVertexAttribPointer (static_cast<GLuint> (attribute_v_color), 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray (static_cast<GLuint> (attribute_coord2d));
  glBindBuffer (GL_ARRAY_BUFFER, vbo_vertices);
  glVertexAttribPointer (static_cast<GLuint> (attribute_coord2d), 2, GL_FLOAT, GL_FALSE, 0, 0);
  glDrawArrays (GL_QUADS, 0, static_cast<int> (elements_count) * 4);
  glBindBuffer (GL_ARRAY_BUFFER, 0);
  glDisableVertexAttribArray (static_cast<GLuint> (attribute_coord2d));
  glDisableVertexAttribArray (static_cast<GLuint> (attribute_v_color));
  program->release ();
  painter.endNativePainting ();

  axes.draw (mvp, painter);

  painter.setRenderHint(QPainter::Antialiasing);
  painter.setRenderHint(QPainter::HighQualityAntialiasing);
  painter.end ();
}