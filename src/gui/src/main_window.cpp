//
// Created by egi on 5/11/19.
//

#include <QtWidgets>
#include <QCheckBox>
#include <QComboBox>

#include "gui/include/main_window.h"
#include "gui/include/opengl_widget.h"

main_window::main_window(unsigned int nx, unsigned int ny, float x_size, float y_size,
    std::function<void()> compute_action,
    std::function<void(float *)> render_action)
  : gl (new opengl_widget (nx, ny, x_size, y_size))
  , renderer (gl, compute_action, render_action)
{
  // Set OpenGL Version information
  // Note: This format must be set before show() is called.
  QSurfaceFormat format;
  format.setRenderableType(QSurfaceFormat::OpenGL);

  gl->setFormat(format);

  setCentralWidget (gl);
  connect (&renderer, SIGNAL (steps_completed ()), gl, SLOT (update_colors ()));
  connect (&renderer, SIGNAL (simulation_completed ()), this, SLOT (simulation_completed ()));

  create_actions ();
  statusBar ()->showMessage ("Ready");
}

main_window::~main_window() = default;

void main_window::start_simulation()
{
  run_action->setEnabled (false);
  stop_action->setEnabled (true);
  renderer.render (use_gpu->isChecked (), gpu_names->currentData ().toInt ());
}

void main_window::simulation_completed ()
{
  run_action->setEnabled (true);
  stop_action->setEnabled (false);
}

void main_window::halt_simulation()
{
  renderer.halt ();

  run_action->setEnabled (true);
  stop_action->setEnabled (false);
}

#ifdef GPU_BUILD
#include <cuda_runtime.h>
#endif

void main_window::create_actions()
{
  QToolBar *control_tool_bar = addToolBar ("Test");

  const QIcon run_icon = style ()->standardIcon (QStyle::SP_MediaPlay);
  run_action = new QAction (run_icon, "Run");
  run_action->setStatusTip ("Run simulation");

  const QIcon stop_icon = style ()->standardIcon (QStyle::SP_MediaPause);
  stop_action = new QAction (stop_icon, "Stop");
  stop_action->setStatusTip ("Stop simulation");
  stop_action->setEnabled (false);

  control_tool_bar->addAction (stop_action);
  control_tool_bar->addAction (run_action);

#ifdef GPU_BUILD
  use_gpu = new QCheckBox ("Use GPU");
  use_gpu->setChecked (true);
  use_gpu->setLayoutDirection (Qt::RightToLeft);

  gpu_names = new QComboBox ();

  int gpus_count = 0;
  cudaGetDeviceCount (&gpus_count);

  for (int gpu_id = 0; gpu_id < gpus_count; gpu_id++)
  {
    cudaDeviceProp device;
    cudaGetDeviceProperties (&device, gpu_id);
    gpu_names->addItem (device.name, QVariant (gpu_id));
  }

  control_tool_bar->addSeparator ();
  control_tool_bar->addWidget (use_gpu);
  control_tool_bar->addWidget (gpu_names);
#endif

  connect (run_action, SIGNAL (triggered ()), this, SLOT (start_simulation ()));
  connect (stop_action, SIGNAL (triggered ()), this, SLOT (halt_simulation ()));
}
