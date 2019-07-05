//
// Created by egi on 5/11/19.
//

#include <QtWidgets>
#include <QCheckBox>
#include <QComboBox>
#include <QSplitter>

#include "settings/global_parameters_widget.h"
#include "settings/source_settings_widget.h"
#include "core/sm/simulation_manager.h"
#include "core/sm/result_extractor.h"

#include "io/hdf5/hdf5_writer.h"

#include "main_window.h"
#include "settings_widget.h"
#include "graphics_widget.h"
#include "opengl_widget.h"
#include "model_widget.h"

main_window::main_window (project_manager &pm_arg)
  : pm (pm_arg)
  , settings (new settings_widget (pm))
  , graphics (new graphics_widget ())
  , model (new model_widget (pm_arg))
  , cpu_visualizer (new hybrid_results_visualizer (pm))
  , hdf5_dump (new hdf5_writer ("output", pm))
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

  python = new QTextEdit ();
  tabs = new QTabWidget ();
  tabs->addTab (python, "Python");

  auto graphics_and_tabs = new QSplitter (Qt::Orientation::Vertical);
  graphics_and_tabs->addWidget (graphics);
  graphics_and_tabs->addWidget (tabs);
  graphics_and_tabs->setStretchFactor (0, 9);
  graphics_and_tabs->setStretchFactor (1, 1);

  python->setText (QString::fromStdString (pm.get_initializer_script ()));

  auto splitter = new QSplitter ();
  splitter->addWidget (model);
  splitter->addWidget (settings);
  splitter->addWidget (graphics_and_tabs);
  splitter->setStretchFactor (0, 1);
  splitter->setStretchFactor (1, 1);
  splitter->setStretchFactor (2, 9);
  setCentralWidget (splitter);

  auto apply_script = new QAction (this);
  apply_script->setShortcut (Qt::Key_R | Qt::CTRL);
  addAction (apply_script);

  connect (model, SIGNAL (configuration_node_selected (std::size_t)), settings, SLOT (setup_configuration_node (std::size_t)));

  connect (&renderer, SIGNAL (steps_completed (bool)), graphics->gl, SLOT (update_colors (bool)));
  connect (&renderer, SIGNAL (simulation_completed ()), this, SLOT (simulation_completed ()));

  connect (this, SIGNAL (on_close ()), this, SLOT (halt_simulation ()));
  connect (graphics->gl, SIGNAL (widget_is_ready ()), this, SLOT (update_project ()));
  connect (apply_script, SIGNAL (triggered ()), this, SLOT (update_project ()));

  create_actions ();

  pm.append_extractor (cpu_visualizer.get ());
  // pm.append_extractor (hdf5_dump.get ());
  hdf5_dump->open();

  statusBar ()->showMessage ("Ready");
}

main_window::~main_window() = default;

void main_window::update_project ()
{
  pm.set_initializer_script (python->toPlainText ().toStdString ());

  if (use_gpu && gpu_names)
    pm.set_gpu_num (use_gpu->isChecked () ? gpu_names->currentData ().toInt () : -1);

  pm.update_project ();
  graphics->gl->update_project (pm);

  auto &grid = pm.get_grid ();
  auto first_field = grid.get_fields_names ().front ();
  cpu_visualizer->set_target (first_field, graphics->gl->get_colors (pm.get_use_gpu ()));
  renderer.extract ();
}

void main_window::start_simulation()
{
  run_action->setEnabled (false);
  stop_action->setEnabled (true);

  update_project ();
  renderer.render ();
}

void main_window::closeEvent (QCloseEvent *event)
{
  emit on_close ();
  renderer.wait ();
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

void main_window::set_use_gpu (bool)
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
