//
// Created by egi on 6/1/19.
//

#ifndef ANYSIM_GRAPHICS_WIDGET_H
#define ANYSIM_GRAPHICS_WIDGET_H

#include <QWidget>
#include <QTreeView>

class opengl_widget;

class graphics_widget : public QWidget
{
  Q_OBJECT

public:
  graphics_widget ();

  opengl_widget *gl;
};

#endif //ANYSIM_GRAPHICS_WIDGET_H
