//
// Created by egi on 5/11/19.
//

#ifndef FDTD_MAIN_WINDOW_H
#define FDTD_MAIN_WINDOW_H

#include <QMainWindow>

#include <functional>

#include "gui/render_thread.h"

class opengl_widget;

class main_window : public QMainWindow
{
  Q_OBJECT

public:
  main_window () = delete;
  main_window (unsigned int nx, unsigned int ny, float x_size, float y_size,
      std::function<void(GLfloat *)> render_action);
  ~main_window () override;

private slots:
  void start_simulation ();

private:
  void create_actions ();

private:
  opengl_widget *gl;
  QAction *run_action = nullptr;
  QAction *stop_action = nullptr;
  render_thread renderer;
};

#endif //FDTD_MAIN_WINDOW_H
