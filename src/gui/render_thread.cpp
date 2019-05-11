//
// Created by egi on 5/11/19.
//

#include "render_thread.h"

render_thread::render_thread(GLfloat *colors_arg, std::function<void(GLfloat *)> render_action_arg, QObject *parent)
  : QThread (parent)
  , colors (colors_arg)
  , render_action (std::move (render_action_arg))
{ }

render_thread::~render_thread() = default;

void render_thread::render()
{
  if (!isRunning ())
    start (LowPriority); ///< Caller thread starts new thread (run)
}

void render_thread::run()
{
  for (unsigned int i = 0; i < 9000; i++)
  {
    render_action (colors);
    emit rendered ();

    msleep (10);
  }
}