//
// Created by egi on 6/1/19.
//

#include "graphics_widget.h"

#include <QVBoxLayout>
#include <QLabel>

#include "opengl_widget.h"

graphics_widget::graphics_widget ()
    : gl (new opengl_widget ())
{
  auto main_layout = new QVBoxLayout ();
  auto widget_label = new QLabel ("Graphics");

  main_layout->addWidget (widget_label);
  main_layout->addWidget (gl);

  setLayout (main_layout);
}
