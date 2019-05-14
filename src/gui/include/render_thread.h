//
// Created by egi on 5/11/19.
//

#ifndef FDTD_RENDER_THREAD_H
#define FDTD_RENDER_THREAD_H

#include <QThread>

#include <functional>
#include <mutex>

#include "gui_simulation_manager.h"

class opengl_widget;


class render_thread : public QThread
{
  Q_OBJECT

public:
  render_thread (opengl_widget *gl_arg, compute_action_type, render_action_type, QObject *parent = nullptr);
  ~render_thread () override;

  void render (bool use_gpu, int gpu_num);
  void halt ();

signals:
  void steps_completed (bool use_gpu);
  void simulation_completed ();

protected:
  void run () override;

protected:
  std::mutex lock;
  bool halt_execution = false;

  bool use_gpu = false;
  opengl_widget *gl = nullptr;
  compute_action_type compute_action;
  render_action_type render_action;
};

#endif //FDTD_RENDER_THREAD_H
