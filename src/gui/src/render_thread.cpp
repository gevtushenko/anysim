//
// Created by egi on 5/11/19.
//

#include "render_thread.h"
#include "opengl_widget.h"

#include <iostream>

render_thread::render_thread (
    opengl_widget *gl_arg,
    project_manager *pm_arg,
    QObject *parent)
  : QThread (parent)
  , gl (gl_arg)
  , pm (pm_arg)
{ }

render_thread::~render_thread() = default;

void render_thread::render ()
{
  halt_execution = false;

  if (!isRunning ())
    start (LowPriority); ///< Caller thread starts new thread (run)
}

void render_thread::halt ()
{
  std::lock_guard guard (lock);
  halt_execution = true;
}

void render_thread::run()
{
  bool use_gpu = pm->get_use_gpu ();

  for (unsigned int i = 0; i < 4000; i++)
  {
    pm->calculate (10);
    pm->render_function (gl->get_colors (use_gpu));
    emit steps_completed (use_gpu);

    {
      std::lock_guard guard (lock);

      if (halt_execution)
        break;
    }
  }

  emit simulation_completed ();
}