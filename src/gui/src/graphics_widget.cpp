//
// Created by egi on 6/1/19.
//

#include "graphics_widget.h"

#include <QVBoxLayout>
#include <QLabel>

#include "opengl_widget.h"
#include "section_name.h"

graphics_widget::graphics_widget ()
    : gl (new opengl_widget ())
{
  auto main_layout = new QVBoxLayout ();
  auto widget_label = new section_name ("Graphics");

  main_layout->addWidget (widget_label);
  main_layout->addWidget (gl);

  setLayout (main_layout);
}
