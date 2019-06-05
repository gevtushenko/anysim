//
// Created by egi on 5/11/19.
//

#include <QtWidgets>
#include <QCheckBox>
#include <QComboBox>

#include "main_window.h"
#include "settings_widget.h"
#include "graphics_widget.h"
#include "opengl_widget.h"
#include "model_widget.h"

main_window::main_window ()
  : model (new model_widget ())
  , settings (new settings_widget ())
  , graphics (new graphics_widget ())
  , pm (new project_manager (true))
  , renderer (graphics->gl, pm.get ())
{
  // Set OpenGL Version information
  // Note: This format must be set before show() is called.
  QSurfaceFormat format;
  format.setRenderableType(QSurfaceFormat::OpenGL);

  graphics->gl->setFormat(format);
  graphics->gl->setSizePolicy (QSizePolicy::Expanding, QSizePolicy::Expanding);
  model->setSizePolicy (QSizePolicy::Minimum, QSizePolicy::Expanding);

  settings->setHidden (true);

  connect (settings, SIGNAL (source_ready (double, double, double)), this, SLOT (create_source (double, double, double)));
  connect (model, SIGNAL (create_source ()), this, SLOT (initialize_source_creation ()));

  auto layout = new QHBoxLayout ();
  layout->addWidget (model, 1);
  layout->addWidget (settings, 1);
  layout->addWidget (graphics, 3);

  auto central_widget = new QWidget ();
  central_widget->setLayout (layout);
  setCentralWidget (central_widget);

  connect (&renderer, SIGNAL (steps_completed (bool)), graphics->gl, SLOT (update_colors (bool)));
  connect (&renderer, SIGNAL (simulation_completed ()), this, SLOT (simulation_completed ()));

  connect (this, SIGNAL (on_close ()), this, SLOT (halt_simulation ()));
  connect (this, SIGNAL (on_close ()), graphics->gl, SLOT (on_close ()));

  create_actions ();

  statusBar ()->showMessage ("Ready");

  pm->set_use_gpu (false);
}

main_window::~main_window() = default;

void main_window::initialize_source_creation ()
{
  settings->show ();
}

void main_window::create_source (double x, double y, double frequency)
{
  pm->append_source (frequency, x, y);
  settings->hide ();
}

void main_window::start_simulation()
{
  run_action->setEnabled (false);
  stop_action->setEnabled (true);

  pm->prepare_simulation ();
  graphics->gl->update_project (pm.get ());

  // use_gpu ? use_gpu->isChecked () : false, gpu_names ? gpu_names->currentData ().toInt () : 0
  renderer.render ();
}

void main_window::closeEvent (QCloseEvent *event)
{
  emit on_close ();
  event->accept ();
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

void main_window::set_use_gpu (bool checked)
{
  pm->set_use_gpu (checked);
}

void main_window::create_actions()
{
  QToolBar *control_tool_bar = addToolBar ("Test");

  run_action = new QAction (QIcon (":/icons/play.svg"), "Run");
  run_action->setStatusTip ("Run simulation");

  stop_action = new QAction (QIcon (":/icons/pause.svg"), "Stop");
  stop_action->setStatusTip ("Stop simulation");
  stop_action->setEnabled (false);

  control_tool_bar->addAction (stop_action);
  control_tool_bar->addAction (run_action);

#ifdef GPU_BUILD
  use_gpu = new QCheckBox ("Use GPU");
  use_gpu->setLayoutDirection (Qt::RightToLeft);
  use_gpu->setChecked (false);

  connect (use_gpu, SIGNAL (toggled (bool)), this, SLOT (set_use_gpu (bool)));

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
