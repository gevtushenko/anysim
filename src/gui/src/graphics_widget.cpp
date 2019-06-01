//
// Created by egi on 6/1/19.
//

#include "graphics_widget.h"

#include <QVBoxLayout>
#include <QLabel>

#include "opengl_widget.h"

graphics_widget::graphics_widget (
    unsigned int nx, unsigned int ny,
    float x_size, float y_size)
    : gl (new opengl_widget (nx, ny, x_size, y_size))
{
  auto main_layout = new QVBoxLayout ();
  auto widget_label = new QLabel ("Graphics");

  main_layout->addWidget (widget_label);
  main_layout->addWidget (gl);

  setLayout (main_layout);
}
