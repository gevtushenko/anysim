//
// Created by egi on 5/11/19.
//

#ifndef FDTD_RENDER_THREAD_H
#define FDTD_RENDER_THREAD_H

#include <QThread>

#include <functional>
#include <mutex>

#include "core/pm/project_manager.h"
#include "gui_application.h"

class opengl_widget;

class render_thread : public QThread
{
  Q_OBJECT

public:
  render_thread (opengl_widget *gl_arg, project_manager *pm, QObject *parent = nullptr);
  ~render_thread () override;

  void render ();
  void halt ();

signals:
  void steps_completed (bool use_gpu);
  void simulation_completed ();

protected:
  void run () override;

protected:
  std::mutex lock;
  bool halt_execution = false;

  opengl_widget *gl = nullptr;
  project_manager *pm = nullptr;
};

#endif //FDTD_RENDER_THREAD_H
