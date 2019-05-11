//
// Created by egi on 5/11/19.
//

#include "opengl_widget.h"

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

opengl_widget::opengl_widget (unsigned int nx, unsigned int ny)
  : elements_count (nx * ny)
{

}

opengl_widget::~opengl_widget ()
{
  glDeleteBuffers (1, &vbo_vertices);
  glDeleteBuffers (1, &vbo_colors);
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
  const int vertices_per_element = 4;
  const int coords_per_vertex = 2;
  const int vertex_data_per_element = vertices_per_element * coords_per_vertex;
  const long int vertices_array_size = elements_count * vertex_data_per_element * glfloat_size;
  glGenBuffers (1, &vbo_vertices);
  glBindBuffer (GL_ARRAY_BUFFER, vbo_vertices);
  glBufferData (GL_ARRAY_BUFFER, vertices_array_size, vertices.get (), GL_DYNAMIC_DRAW);

  const int colors_per_vertex = 3;
  const int color_data_per_element = colors_per_vertex * vertices_per_element;
  const long int colors_array_size = elements_count * color_data_per_element * glfloat_size;
  glGenBuffers (1, &vbo_colors);
  glBindBuffer (GL_ARRAY_BUFFER, vbo_colors);
  glBufferData (GL_ARRAY_BUFFER, colors_array_size, colors.get (), GL_DYNAMIC_DRAW);
}

void opengl_widget::resizeGL(int width, int height)
{
  (void) width;
  (void) height;
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
}