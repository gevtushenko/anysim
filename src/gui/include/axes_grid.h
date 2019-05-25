//
// Created by egi on 5/23/19.
//

#ifndef ANYSIM_AXES_GRID_H
#define ANYSIM_AXES_GRID_H

#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>

#include <memory>

class axes_grid : protected QOpenGLFunctions
{
public:
  axes_grid () = default;

  void init (
      unsigned int x_tics_arg, unsigned int y_tics,
      float left_x, float right_x,
      float bottom_y, float top_y);
  void draw (QMatrix4x4 &mvp);

private:
  unsigned int x_tics = 0, y_tics = 0;
  unsigned int total_coords = 0;
  std::unique_ptr<GLfloat[]> coords;

  QOpenGLBuffer grid_vbo;
  QOpenGLShaderProgram program;
  QOpenGLVertexArrayObject grid_vao;
};

#endif //ANYSIM_AXES_GRID_H
