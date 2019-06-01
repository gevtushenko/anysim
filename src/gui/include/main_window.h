//
// Created by egi on 5/11/19.
//

#ifndef FDTD_MAIN_WINDOW_H
#define FDTD_MAIN_WINDOW_H

#include <QMainWindow>

#include <functional>

#include "render_thread.h"

class QCheckBox;
class QComboBox;

class graphics_widget;
class model_widget;

class main_window : public QMainWindow
{
  Q_OBJECT

public:
  main_window () = delete;
  main_window (unsigned int nx, unsigned int ny, float x_size, float y_size,
      compute_action_type, render_action_type);
  ~main_window () override;

private slots:
  void start_simulation ();
  void simulation_completed ();
  void halt_simulation ();

private:
  void create_actions ();

private:
  model_widget *model;
  graphics_widget *graphics;

  QAction *run_action = nullptr;
  QAction *stop_action = nullptr;
  QCheckBox *use_gpu = nullptr;
  QComboBox *gpu_names = nullptr;
  render_thread renderer;
};

#endif //FDTD_MAIN_WINDOW_H
