//
// Created by egi on 5/26/19.
//

#include "camera.h"

#include "cpp/common_funcs.h"

camera::camera()
{
  reset ();
  orthographic_projection.setToIdentity ();
}

void camera::reset ()
{
  model.setToIdentity ();
  scale.setToIdentity ();
  rotation.setToIdentity ();
  translation.setToIdentity ();
}

QMatrix4x4 camera::get_mvp ()
{
  return translation * rotation * scale * orthographic_projection * model;
}

void camera::resize (int width, int height)
{
  view_width = width;
  view_height = height;

  float aspect = static_cast<float> (view_width) / view_height;

  calculate_orthographic_projection (
      -aspect /* left*/, aspect /* right */,
      -1.0 /* bottom */, 1.0 /* top */,
      -1.0 /* near */, 1.0 /* far */,
      orthographic_projection);
}

void camera::update_model_matrix (float width, float height)
{
  const float quarter_size = 0.9;
  const float delim = std::max (width, height);
  model (0, 0) = 2.0 * quarter_size / delim;
  model (1, 1) = 2.0 * quarter_size / delim;
  model (0, 3) = -quarter_size;
  model (1, 3) = -quarter_size;
}

void camera::zoom (int wheel_delta)
{
  if (wheel_delta != 0)
    update_scaling_matrix (std::pow (0.9f, -static_cast<float> (wheel_delta) / 120));
}

void camera::move (int dx, int dy)
{
  translation (0, 3) -= static_cast<float> (dx) / (static_cast<float> (view_width) / 2);
  translation (1, 3) += static_cast<float> (dy) / (static_cast<float> (view_height) / 2);
}

void camera::update_scaling_matrix (float scaling_coefficient)
{
  scale.scale (scaling_coefficient);
}

void camera::calculate_orthographic_projection (
    float left, float right,
    float bottom, float top,
    float near, float far,
    QMatrix4x4 &projection)
{
  projection (0, 0) =  2.0 / (right - left);
  projection (1, 1) =  2.0 / (top - bottom);
  projection (2, 2) = -2.0 / (far - near);

  projection (3, 0) = -static_cast<float> (right + left) / (right - left);
  projection (3, 1) = -static_cast<float> (top + bottom) / (top - bottom);
  projection (3, 2) = -static_cast<float> (far + near) / (far - near);
}
