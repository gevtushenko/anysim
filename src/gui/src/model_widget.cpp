//
// Created by egi on 6/1/19.
//

#include "model_widget.h"

#include <QVBoxLayout>
#include <QLabel>

model_widget::model_widget ()
{
  auto main_layout = new QVBoxLayout ();
  auto widget_label = new QLabel ("Model");

  view = new QTreeView ();

  main_layout->addWidget (widget_label);
  main_layout->addWidget (view);

  setLayout (main_layout);
}