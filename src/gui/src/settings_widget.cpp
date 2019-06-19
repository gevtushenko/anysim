//
// Created by egi on 6/3/19.
//

#include "settings_widget.h"
#include "settings/global_parameters_widget.h"
#include "settings/source_settings_widget.h"


#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>

settings_widget::settings_widget ()
{
  auto main_layout = new QVBoxLayout ();
  auto widget_label = new QLabel ("Settings");

  source_widget = new source_settings_widget ();
  global_params_widget = new global_parameters_widget ();

  source_widget->hide ();
  global_params_widget->hide ();

  main_layout->addWidget (widget_label);
  main_layout->addWidget (source_widget);
  main_layout->addWidget (global_params_widget);
  main_layout->addStretch (1);

  setLayout (main_layout);
}

