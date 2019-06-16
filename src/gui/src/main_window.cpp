//
// Created by egi on 5/11/19.
//

#include <QtWidgets>
#include <QCheckBox>
#include <QComboBox>

#include "settings/global_parameters_widget.h"
#include "settings/source_settings_widget.h"
#include "core/sm/simulation_manager.h"
#include "core/sm/result_extractor.h"

#include "main_window.h"
#include "settings_widget.h"
#include "graphics_widget.h"
#include "opengl_widget.h"
#include "model_widget.h"

main_window::main_window (project_manager &pm_arg)
  : pm (pm_arg)
  , settings (new settings_widget ())
  , graphics (new graphics_widget ())
  , model (new model_widget (settings))
  , renderer (graphics->gl, &pm)
{
  // Set OpenGL Version information
  // Note: This format must be set before show() is called.
  QSurfaceFormat format;
  format.setRenderableType(QSurfaceFormat::OpenGL);

  graphics->gl->setFormat(format);
  graphics->gl->setSizePolicy (QSizePolicy::Expanding, QSizePolicy::Expanding);
  model->setSizePolicy (QSizePolicy::Minimum, QSizePolicy::Expanding);

  settings->setHidden (true);

  connect (settings->source_widget, SIGNAL (source_ready (double, double, double)), this, SLOT (create_source (double, double, double)));
  connect (settings->global_params_widget, SIGNAL (cells_per_lambda_changed (unsigned int)), this, SLOT (update_cells_per_lambda (unsigned int)));

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
}

main_window::~main_window() = default;

void main_window::create_source (double , double , double )
{
  // pm.append_source (frequency, x, y);
  settings->hide ();
}


void main_window::update_cells_per_lambda (unsigned int )
{
  settings->hide ();
}

void main_window::start_simulation()
{
  run_action->setEnabled (false);
  stop_action->setEnabled (true);

  graphics->gl->update_project (&pm);

  cpu_visualizer = std::make_unique<cpu_results_visualizer> (pm.get_solver_workspace (), 200, 100, graphics->gl->get_colors (false));
  cpu_visualizer->set_target ("rho");
  pm.get_simulation_manager ().append_extractor (cpu_visualizer.get ());

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

void main_window::set_use_gpu (bool )
{
}

void main_window::create_actions()
{
  QToolBar *control_tool_bar = addToolBar ("Test");

  run_action = new QAction (QIcon (":/icons/play.svg"), "Run", this);
  run_action->setStatusTip ("Run simulation");

  stop_action = new QAction (QIcon (":/icons/pause.svg"), "Stop", this);
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
