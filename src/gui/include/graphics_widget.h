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
  graphics_widget (
    unsigned int nx, unsigned int ny,
    float x_size, float y_size);

  opengl_widget *gl;
};

#endif //ANYSIM_GRAPHICS_WIDGET_H
