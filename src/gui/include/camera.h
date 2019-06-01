//
// Created by egi on 5/26/19.
//

#ifndef ANYSIM_CAMERA_H
#define ANYSIM_CAMERA_H

#include <QVector3D>
#include <QVector4D>
#include <QMatrix4x4>
#include <QtMath>

class camera
{
public:
  camera ();

  void resize (int width, int height);
  void zoom (int wheel_delta);

  void reset ();

  QMatrix4x4 get_mvp ();

public:
  QMatrix4x4 scale, rotation, translation;
  QMatrix4x4 orthographic_projection;

private:
  void update_scaling_matrix (float scaling_coefficient);
  void calculate_orthographic_projection (
      float left, float right,
      float bottom, float top,
      float near, float far);
};

#endif //ANYSIM_CAMERA_H
