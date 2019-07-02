//
// Created by egi on 5/23/19.
//

#include "axes_grid.h"
#include "cpp/common_funcs.h"
#include "core/gpu/coloring.cuh"

#include <QPainter>

axes_grid::axes_grid () : QOpenGLFunctions (), font ("Arial", 10)
{

}

void axes_grid::initialize_gl (QObject *parent)
{
  initializeOpenGLFunctions ();

  program = new QOpenGLShaderProgram (parent);
  if (!program->isLinked ())
  {
    program->addShaderFromSourceFile (QOpenGLShader::Vertex,   ":/shaders/axes_grid.vert");
    program->addShaderFromSourceFile (QOpenGLShader::Fragment, ":/shaders/axes_grid.frag");
    program->link ();
  }

  attribute_coord2d = program->attributeLocation ("coord2d");
  glGenBuffers (1, &vbo_vertices);
}

void axes_grid::prepare (
  float left_x_arg,
  float right_x_arg,
  float bottom_y_arg,
  float top_y_arg)
{
  top_y = top_y_arg;
  left_x = left_x_arg;
  right_x = right_x_arg;
  bottom_y = bottom_y_arg;
  x_size = (right_x - left_x);
  y_size = (top_y - bottom_y);
}

void axes_grid::init_data ()
{
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

  glBindBuffer (GL_ARRAY_BUFFER, vbo_vertices);
  glBufferData (GL_ARRAY_BUFFER, sizeof (GLfloat) * total_coords, coords.get (), GL_DYNAMIC_DRAW);
}

void axes_grid::resize (int window_width_arg, int window_height_arg)
{
  window_width = window_width_arg;
  window_height = window_height_arg;
}

void axes_grid::draw (const QMatrix4x4 &mvp, QPainter &painter)
{
  QFontMetrics fm (font);
  painter.beginNativePainting ();

  const char *format = "%.2e";

  const double x_max_number = x_size;

  QVector4D lbc (left_x, bottom_y, 0.0, 1.0);
  QVector4D rtc (right_x, top_y, 0.0, 1.0);

  lbc = mvp * lbc;
  rtc = mvp * rtc;

  // TODO Get ticks count
  unsigned int new_tick_count_x = 1;
  unsigned int new_tick_count_y = 1;

  {
    int size = std::snprintf (nullptr, 0, format, x_max_number);
    buf.resize (size + 1);
    std::snprintf (buf.data (), buf.size (), format, x_max_number);

    const int max_text_height = fm.height ();
    const int max_text_width = fm.width (QString::fromStdString (buf.data ()));

    const float right_bottom_corner_x = map (rtc.x (), -1.0, 1.0, 0.0, static_cast<float> (window_width));
    const float right_bottom_corner_y = map (lbc.y (), -1.0, 1.0, 0.0, static_cast<float> (window_height));

    const float left_top_corner_x = map (lbc.x (), -1.0, 1.0, 0.0, static_cast<float> (window_width));
    const float left_top_corner_y = map (rtc.y (), -1.0, 1.0, 0.0, static_cast<float> (window_height));

    const int screen_width = (right_bottom_corner_x - left_top_corner_x);
    const int screen_height = (left_top_corner_y - right_bottom_corner_y);

    const int text_fits_y = screen_height / max_text_height;
    const int text_fits_x = screen_width / max_text_width;

    new_tick_count_x = long_tic_each * text_fits_x / 2;
    new_tick_count_y = long_tic_each * text_fits_y / 2;
  }

  if (new_tick_count_x != x_tics || new_tick_count_y != y_tics)
  {
    x_tics = new_tick_count_x;
    y_tics = new_tick_count_y;

    init_data ();
  }

  program->bind();
  program->setUniformValue ("MVP", mvp);

  glEnableVertexAttribArray (attribute_coord2d);
  glBindBuffer (GL_ARRAY_BUFFER, vbo_vertices);
  glVertexAttribPointer (attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glLineWidth (2.2);
  glDrawArrays (GL_LINES, 0, total_coords);
  glDisableVertexAttribArray (attribute_coord2d);
  glBindBuffer (GL_ARRAY_BUFFER, 0);

  program->release ();
  painter.endNativePainting ();

  painter.setPen(Qt::black);
  painter.setFont(font);

  const float dx = x_size / (x_tics - 1);
  const float dy = y_size / (y_tics - 1);

  float *p_coords = coords.get ();

  for (unsigned int y = 0; y < y_tics; y+=long_tic_each)
  {
    const double number = y * dy;
    const int size = std::snprintf (nullptr, 0, format, number);
    buf.resize (size + 1);
    std::snprintf (buf.data (), buf.size (), format, number);

    QVector4D v (p_coords[2] - long_tic_size / 4, p_coords[3], 0.0, 1.0);
    v = mvp * v;

    const int right_bottom_corner_x = map (v.x (), -1.0, 1.0, 0.0, static_cast<float> (window_width));
    const int right_bottom_corner_y = map (v.y (),  1.0, -1.0, 0.0, static_cast<float> (window_height)) + fm.height () / 2;

    painter.drawText (QRect (0, 0, right_bottom_corner_x, right_bottom_corner_y), Qt::AlignRight | Qt::AlignBottom, QString::fromStdString (buf.data ()));
    p_coords += 8 * long_tic_each;
  }
  p_coords = coords.get () + 8 * y_tics;
  for (unsigned int x = 0; x < x_tics; x+=long_tic_each)
  {
    const double number = x * dx;
    const int size = std::snprintf (nullptr, 0, format, number);
    buf.resize (size + 1);
    std::snprintf (buf.data (), buf.size (), format, number);

    QVector4D v (p_coords[2], p_coords[3] - long_tic_size / 4, 0.0, 1.0);
    v = mvp * v;

    const int right_bottom_corner_x = map (v.x (), -1.0, 1.0, 0.0, static_cast<float> (window_width));
    const int right_bottom_corner_y = map (v.y (),  1.0, -1.0, 0.0, static_cast<float> (window_height));

    painter.drawText (QRect (right_bottom_corner_x, right_bottom_corner_y, window_width, window_height), Qt::AlignLeft | Qt::AlignTop, QString::fromStdString (buf.data ()));
    p_coords += 8 * long_tic_each;
  }
}
