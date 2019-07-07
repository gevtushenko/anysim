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
class QTextEdit;

class python_syntax_highlighter;
class hybrid_results_visualizer;
class graphics_widget;
class settings_widget;
class project_manager;
class model_widget;

class main_window : public QMainWindow
{
  Q_OBJECT

public:
  main_window () = delete;
  explicit main_window (project_manager &pm_arg);
  ~main_window () override;

private slots:
  void update_project ();
  void start_simulation ();
  void simulation_completed ();
  void halt_simulation ();
  void set_use_gpu (bool checked);

signals:
  void on_close ();

private:
  void create_actions ();

  void closeEvent (QCloseEvent *event) override;

private:
  project_manager &pm;
  settings_widget *settings;
  graphics_widget *graphics;
  model_widget *model;

  std::unique_ptr<hybrid_results_visualizer> cpu_visualizer;

  QAction *run_action = nullptr;
  QAction *stop_action = nullptr;
  QCheckBox *use_gpu = nullptr;
  QComboBox *gpu_names = nullptr;
  QTextEdit *python = nullptr;
  python_syntax_highlighter *highlighter = nullptr;
  QTabWidget *tabs = nullptr;
  render_thread renderer;
};

#endif //FDTD_MAIN_WINDOW_H
