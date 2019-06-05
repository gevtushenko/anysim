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
class settings_widget;
class project_manager;
class model_widget;

class main_window : public QMainWindow
{
  Q_OBJECT

public:
  main_window ();
  ~main_window () override;

private slots:
  void start_simulation ();
  void simulation_completed ();
  void halt_simulation ();
  void initialize_source_creation ();
  void create_source (double x, double y, double frequency);
  void set_use_gpu (bool checked);

signals:
  void on_close ();

private:
  void create_actions ();

  void closeEvent (QCloseEvent *event) override;

private:
  model_widget *model;
  settings_widget *settings;
  graphics_widget *graphics;

  std::unique_ptr<project_manager> pm;

  QAction *run_action = nullptr;
  QAction *stop_action = nullptr;
  QCheckBox *use_gpu = nullptr;
  QComboBox *gpu_names = nullptr;
  render_thread renderer;
};

#endif //FDTD_MAIN_WINDOW_H
