//
// Created by egi on 5/23/19.
//

#include "axes_grid.h"
#include "text_renderer.h"
#include "cpp/common_funcs.h"

void axes_grid::init (
    unsigned int x_tics_arg, unsigned int y_tics_arg,
    float left_x, float right_x,
    float bottom_y, float top_y)
{
  initializeOpenGLFunctions ();

  x_tics = x_tics_arg; y_tics = y_tics_arg;

  if (!program.isLinked ())
  {
    program.addShaderFromSourceFile (QOpenGLShader::Vertex,   ":/shaders/axes_grid.vert");
    program.addShaderFromSourceFile (QOpenGLShader::Fragment, ":/shaders/axes_grid.frag");
    program.link ();
  }

  grid_vao.create();
  grid_vao.bind();

  const unsigned int points_per_line = 2;
  const unsigned int coords_per_point = 2;
  const unsigned int coords_per_oxs = 2 * x_tics * points_per_line * coords_per_point;
  const unsigned int coords_per_oys = 2 * y_tics * points_per_line * coords_per_point;
  const unsigned int border_coords = 8 * coords_per_point;
  const unsigned int tiks_coords = coords_per_oxs + coords_per_oys;
  total_coords = border_coords + tiks_coords;

  coords.reset ();
  coords = std::make_unique<GLfloat[]> (total_coords);

  coords[tiks_coords + 0]  = left_x;  coords[tiks_coords + 1]  = bottom_y;
  coords[tiks_coords + 2]  = left_x;  coords[tiks_coords + 3]  = top_y;

  coords[tiks_coords + 4]  = left_x;  coords[tiks_coords + 5]  = top_y;
  coords[tiks_coords + 6]  = right_x; coords[tiks_coords + 7]  = top_y;

  coords[tiks_coords + 8]  = right_x; coords[tiks_coords + 9]  = top_y;
  coords[tiks_coords + 10] = right_x; coords[tiks_coords + 11] = bottom_y;

  coords[tiks_coords + 12] = right_x; coords[tiks_coords + 13] = bottom_y;
  coords[tiks_coords + 14] = left_x;  coords[tiks_coords + 15] = bottom_y;

  float dy = (top_y - bottom_y) / (y_tics - 1);
  float *p_coords = coords.get ();

  float tic_size = 0.01f;
  float long_tic_size = 0.025f;
  const unsigned int long_tic_each = 4;

  const bool in = false; /// Tics are inside model
  const short int dir = in ? -1 : 1;

  for (unsigned int y = 0; y < y_tics; y++)
  {
    const float ts = y % long_tic_each == 0 ? long_tic_size : tic_size;

    p_coords[0] = left_x;
    p_coords[1] = bottom_y + dy * y;
    p_coords[2] = left_x - dir * ts;
    p_coords[3] = bottom_y + dy * y;

    p_coords[4] = right_x;
    p_coords[5] = bottom_y + dy * y;
    p_coords[6] = right_x + dir * ts;
    p_coords[7] = bottom_y + dy * y;

    p_coords += 8;
  }

  float dx = (right_x - left_x) / (x_tics - 1);

  for (unsigned int x = 0; x < x_tics; x++)
  {
    const float ts = x % long_tic_each == 0 ? long_tic_size : tic_size;

    p_coords[0] = left_x + dx * x;
    p_coords[1] = bottom_y;
    p_coords[2] = left_x + dx * x;
    p_coords[3] = bottom_y - dir * ts;

    p_coords[4] = left_x + dx * x;
    p_coords[5] = top_y;
    p_coords[6] = left_x + dx * x;
    p_coords[7] = top_y + dir * ts;

    p_coords += 8;
  }

  grid_vbo.create();
  grid_vbo.bind();
  grid_vbo.allocate(sizeof(GLfloat) * (total_coords));
  grid_vbo.write (0, coords.get (), sizeof (GLfloat) * (total_coords));
  grid_vbo.setUsagePattern(QOpenGLBuffer::DynamicDraw);

  program.setAttributeBuffer ("coord2d", GL_FLOAT, 0, 2);
  program.enableAttributeArray ("coord2d");
  grid_vao.release();
}

void axes_grid::draw (QMatrix4x4 &mvp)
{
  auto &tr = text_renderer::instance ();

  QMatrix4x4 matrix;
  matrix.ortho(QRect(0, 0, 800, 600));
  tr.render_text ("test", 0, 0, 1.0, matrix);

  cpp_unreferenced (mvp);

  // program.bind();
  // program.setUniformValue ("MVP", mvp);

  // grid_vao.bind ();
  // grid_vbo.bind ();
  // glLineWidth (2.2);
  // glDrawArrays (GL_LINES, 0, total_coords);
  // grid_vbo.release ();
  // grid_vao.release ();

  // program.release ();
}
