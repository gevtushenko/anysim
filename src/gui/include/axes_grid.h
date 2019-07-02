//
// Created by egi on 5/23/19.
//

#ifndef ANYSIM_AXES_GRID_H
#define ANYSIM_AXES_GRID_H

#include <QFont>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>

#include <memory>

class QPainter;

class axes_grid : protected QOpenGLFunctions
{
public:
  axes_grid ();

  void initialize_gl (QObject *parent);
  void resize (int window_width_arg, int window_height_arg);
  void prepare (
    float left_x, float right_x,
    float bottom_y, float top_y);
  void draw (const QMatrix4x4 &mvp, QPainter &painter);

private:
  void init_data ();

private:
  QFont font;
  std::vector<char> buf;
  float x_size, y_size;
  float left_x, right_x;
  float bottom_y, top_y;
  const float tic_size = 0.01f;
  const float long_tic_size = 0.025f;
  unsigned int window_width = 1;
  unsigned int window_height = 1;
  unsigned int x_tics = 0, y_tics = 0;
  unsigned int total_coords = 0;
  const unsigned int long_tic_each = 4;
  std::unique_ptr<GLfloat[]> coords;

  GLuint vbo_vertices;
  GLint attribute_coord2d;
  QOpenGLShaderProgram *program;
};

#endif //ANYSIM_AXES_GRID_H
