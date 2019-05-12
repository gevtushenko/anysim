//
// Created by egi on 5/11/19.
//

#include "gui/include/render_thread.h"

#include <iostream>

render_thread::render_thread(GLfloat *colors_arg, std::function<void(GLfloat *)> render_action_arg, QObject *parent)
  : QThread (parent)
  , colors (colors_arg)
  , render_action (std::move (render_action_arg))
{ }

render_thread::~render_thread() = default;

void render_thread::render (bool use_gpu, int gpu_num)
{
  std::cout << "Use gpu: " << use_gpu << ": " << gpu_num << std::endl;

  if (!isRunning ())
    start (LowPriority); ///< Caller thread starts new thread (run)
}

void render_thread::run()
{
  for (unsigned int i = 0; i < 100; i++)
  {
    render_action (colors);
    emit steps_completed ();
    // msleep (10);
  }

  emit simulation_completed ();
}