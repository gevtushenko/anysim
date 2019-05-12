//
// Created by egi on 5/11/19.
//

#include "gui/include/opengl_widget.h"

#include <iostream>

const char *vs_source =
    //"#version 100\n"  // OpenGL ES 2.0
    //"#version 120\n"  // OpenGL 2.1
    "attribute vec2 coord2d;                  "
    "attribute vec3 v_color;                  "
    "varying vec3 f_color;                    "
    "void main(void) {                        "
    "  gl_Position = vec4(coord2d, 0.0, 1.0); "
    "  f_color = v_color;                     "
    "}";

const char *fs_source =
    //"#version 100\n"  // OpenGL ES 2.0
    //"#version 120\n"  // OpenGL 2.1
    "varying highp vec3 f_color;    "
    "void main(void) {        "
    "  gl_FragColor = vec4(f_color.r, f_color.g, f_color.b, 1.0); "
    "}";

opengl_widget::opengl_widget (unsigned int nx, unsigned int ny, float x_size, float y_size)
  : elements_count (nx * ny)
  , colors (new GLfloat[color_data_per_element * elements_count])
  , vertices (new GLfloat[vertex_data_per_element * elements_count])
{
  const GLfloat l_x = -0.8f;
  const GLfloat r_x =  0.8f;
  const GLfloat b_y = -0.8f;
  const GLfloat t_y =  0.8f;

  GLfloat max_width  = (r_x - l_x) * (x_size >= y_size ? 1.0f : x_size / y_size);
  GLfloat max_height = (t_y - b_y) * (y_size >  x_size ? 1.0f : y_size / x_size);

  GLfloat dx = max_width / static_cast<GLfloat> (nx);
  GLfloat dy = max_height / static_cast<GLfloat> (ny);

  static GLfloat _colors[] =
      {
          1.0, 1.0, 0.0,
          0.0, 0.0, 1.0,
          1.0, 0.0, 0.0,
          1.0, 0.0, 1.0,
      };

  for (unsigned int j = 0; j < ny; j++)
  {
    for (unsigned int i = 0; i < nx; i++)
    {
      const unsigned int vert_offset = static_cast<unsigned int> (vertex_data_per_element) * (j * nx + i);

      vertices[vert_offset + 0] = l_x + dx * (i + 0); vertices[vert_offset + 1] = b_y + dy * (j + 1);
      vertices[vert_offset + 2] = l_x + dx * (i + 0); vertices[vert_offset + 3] = b_y + dy * (j + 0);
      vertices[vert_offset + 4] = l_x + dx * (i + 1); vertices[vert_offset + 5] = b_y + dy * (j + 0);
      vertices[vert_offset + 6] = l_x + dx * (i + 1); vertices[vert_offset + 7] = b_y + dy * (j + 1);

      const unsigned int color_offset = static_cast<unsigned int> (color_data_per_element) * (j * nx + i);
      std::copy_n (_colors, color_data_per_element, colors.get () + color_offset);
    }
  }

  std::cout << max_width << " - " << max_height << std::endl;
}

opengl_widget::~opengl_widget ()
{
  glDeleteBuffers (1, &vbo_vertices);
  glDeleteBuffers (1, &vbo_colors);
}

GLfloat *opengl_widget::get_colors ()
{
  return colors.get ();
}

void opengl_widget::initializeGL()
{
  initializeOpenGLFunctions ();
  program = std::make_unique<QOpenGLShaderProgram> (this);
  program->addShaderFromSourceCode (QOpenGLShader::Vertex, vs_source);
  program->addShaderFromSourceCode (QOpenGLShader::Fragment, fs_source);
  program->link ();
  attribute_coord2d = program->attributeLocation ("coord2d");
  attribute_v_color = program->attributeLocation ("v_color");

  /// VBO Handling
  const int glfloat_size = sizeof (GLfloat);
  const long int vertices_array_size = elements_count * vertex_data_per_element * glfloat_size;
  glGenBuffers (1, &vbo_vertices);
  glBindBuffer (GL_ARRAY_BUFFER, vbo_vertices);
  glBufferData (GL_ARRAY_BUFFER, vertices_array_size, vertices.get (), GL_DYNAMIC_DRAW);

  const long int colors_array_size = elements_count * color_data_per_element * glfloat_size;
  glGenBuffers (1, &vbo_colors);
  glBindBuffer (GL_ARRAY_BUFFER, vbo_colors);
  glBufferData (GL_ARRAY_BUFFER, colors_array_size, colors.get (), GL_DYNAMIC_DRAW);

  initialized = true;
}

void opengl_widget::resizeGL(int width, int height)
{
  (void) width;
  (void) height;

  if (!initialized)
    return;
}

void opengl_widget::update_colors()
{
  const int glfloat_size = sizeof (GLfloat);
  const long int colors_array_size = elements_count * color_data_per_element * glfloat_size;

  glBindBuffer (GL_ARRAY_BUFFER, vbo_colors);
  glBufferData (GL_ARRAY_BUFFER, colors_array_size, colors.get (), GL_DYNAMIC_DRAW);
  update ();
}

void opengl_widget::paintGL()
{
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  program->bind();
  glEnableVertexAttribArray (static_cast<GLuint> (attribute_v_color));
  glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
  glVertexAttribPointer(static_cast<GLuint> (attribute_v_color), 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(static_cast<GLuint> (attribute_coord2d));
  glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
  glVertexAttribPointer (static_cast<GLuint> (attribute_coord2d), 2, GL_FLOAT, GL_FALSE, 0, 0);
  glDrawArrays(GL_QUADS, 0, static_cast<int> (elements_count) * 4);
  glDisableVertexAttribArray(static_cast<GLuint> (attribute_coord2d));
  glDisableVertexAttribArray(static_cast<GLuint> (attribute_v_color));
  program->release();

  std::cout << "painted" << std::endl;
}