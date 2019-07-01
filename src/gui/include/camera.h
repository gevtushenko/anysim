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
  void move (int dx, int dy);

  void update_model_matrix (float width, float height);

  void reset ();

  QMatrix4x4 get_mvp ();

public:
  int view_width = 0;
  int view_height = 0;

  QMatrix4x4 scale, rotation, translation;
  QMatrix4x4 orthographic_projection, model;

private:
  void update_scaling_matrix (float scaling_coefficient);
  void calculate_orthographic_projection (
      float left, float right,
      float bottom, float top,
      float near, float far,
      QMatrix4x4 &projection);
};

#endif //ANYSIM_CAMERA_H
