//
// Created by egi on 5/11/19.
//

#ifndef FDTD_RENDER_THREAD_H
#define FDTD_RENDER_THREAD_H

#include <QThread>

#include <functional>
#include <mutex>

class opengl_widget;

class render_thread : public QThread
{
  Q_OBJECT

public:
  explicit render_thread (opengl_widget *gl_arg, std::function<void()>, std::function<void(float *)>, QObject *parent = nullptr);
  ~render_thread () override;

  void render (bool use_gpu, int gpu_num);
  void halt ();

signals:
  void steps_completed ();
  void simulation_completed ();

protected:
  void run () override;

protected:
  std::mutex lock;
  bool halt_execution;
  opengl_widget *gl = nullptr;
  std::function<void()> compute_action;
  std::function<void(float *)> render_action;
};

#endif //FDTD_RENDER_THREAD_H
