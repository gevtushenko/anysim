//
// Created by egi on 5/11/19.
//

#ifndef FDTD_MAIN_WINDOW_H
#define FDTD_MAIN_WINDOW_H

#include <QMainWindow>

class opengl_widget;

class main_window : public QMainWindow
{
  Q_OBJECT

public:
  main_window (unsigned int nx, unsigned int ny, float x_size, float y_size);
  ~main_window () override;

  opengl_widget *gl;
};

#endif //FDTD_MAIN_WINDOW_H
