//
// Created by egi on 5/11/19.
//

#include "render_thread.h"
#include "opengl_widget.h"

#include <iostream>

render_thread::render_thread (
    opengl_widget *gl_arg,
    std::function<void()> compute_action_arg,
    std::function<void(float *)> render_action_arg,
    QObject *parent)
  : QThread (parent)
  , gl (gl_arg)
  , compute_action (std::move (compute_action_arg))
  , render_action (std::move (render_action_arg))
{ }

render_thread::~render_thread() = default;

void render_thread::render (bool use_gpu, int gpu_num)
{
  std::cout << "Use gpu: " << use_gpu << ": " << gpu_num << std::endl;

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
  for (unsigned int i = 0; i < 4000; i++)
  {
    compute_action ();
    float *colors = gl->get_gpu_colors ();
    render_action (colors);
    emit steps_completed ();

    {
      std::lock_guard guard (lock);

      if (halt_execution)
        break;
    }
  }

  emit simulation_completed ();
}