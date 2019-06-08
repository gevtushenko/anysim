//
// Created by egi on 6/3/19.
//

#ifndef ANYSIM_SETTINGS_WIDGET_H
#define ANYSIM_SETTINGS_WIDGET_H

#include <QWidget>

class source_settings_widget;
class global_parameters_widget;

class settings_widget : public QWidget
{
Q_OBJECT

public:
  settings_widget ();

public:
  source_settings_widget *source_widget = nullptr;
  global_parameters_widget *global_params_widget = nullptr;

public slots:
  void show_global_parameters ();
  void show_source_settings ();
};

#endif //ANYSIM_SETTINGS_WIDGET_H
