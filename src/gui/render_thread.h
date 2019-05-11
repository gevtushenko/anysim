//
// Created by egi on 5/11/19.
//

#ifndef FDTD_RENDER_THREAD_H
#define FDTD_RENDER_THREAD_H

#include <QThread>
#include <QOpenGLWidget>

#include <functional>

class render_thread : public QThread
{
  Q_OBJECT

public:
  explicit render_thread (GLfloat *colors_arg, std::function<void(GLfloat *)>, QObject *parent = nullptr);
  ~render_thread () override;

  void render ();

signals:
  void rendered ();

protected:
  void run () override;

protected:
  GLfloat *colors;
  std::function<void(GLfloat *)> render_action;
};

#endif //FDTD_RENDER_THREAD_H
