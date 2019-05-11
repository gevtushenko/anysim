//
// Created by egi on 5/11/19.
//

#include <QtWidgets>

#include "main_window.h"
#include "gui/opengl_widget.h"

main_window::main_window(unsigned int nx, unsigned int ny, float x_size, float y_size,
    std::function<void(GLfloat *)> render_action)
  : gl (new opengl_widget (nx, ny, x_size, y_size))
  , renderer (gl->get_colors (), render_action)
{
  // Set OpenGL Version information
  // Note: This format must be set before show() is called.
  QSurfaceFormat format;
  format.setRenderableType(QSurfaceFormat::OpenGL);

  gl->setFormat(format);

  setCentralWidget (gl);
  connect (&renderer, SIGNAL (rendered ()), gl, SLOT (update_colors ()));
  // renderer.render ();

  create_actions ();
  statusBar ()->showMessage ("Ready");
}

main_window::~main_window() = default;

void main_window::start_simulation()
{
  run_action->setEnabled (false);
  renderer.render ();
}

void main_window::create_actions()
{
  QToolBar *control_tool_bar = addToolBar ("Test");

  const QIcon run_icon = style ()->standardIcon (QStyle::SP_MediaPlay);
  run_action = new QAction (run_icon, "Run");
  run_action->setStatusTip ("Run simulation");

  const QIcon stop_icon = style ()->standardIcon (QStyle::SP_MediaStop);
  stop_action = new QAction (stop_icon, "Stop");
  stop_action->setStatusTip ("Stop simulation");
  stop_action->setEnabled (false);

  control_tool_bar->addAction (stop_action);
  control_tool_bar->addAction (run_action);

  connect (run_action, SIGNAL (triggered ()), this, SLOT (start_simulation ()));
}
