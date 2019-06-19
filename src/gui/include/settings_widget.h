//
// Created by egi on 6/3/19.
//

#ifndef ANYSIM_SETTINGS_WIDGET_H
#define ANYSIM_SETTINGS_WIDGET_H

#include <QWidget>

class QVBoxLayout;
class configuration_node;

class settings_widget : public QWidget
{
Q_OBJECT

public:
  settings_widget ();

public slots:
  void setup_configuration_node (configuration_node *node);

private:
  QVBoxLayout *main_layout = nullptr;
  QVBoxLayout *node_layout = nullptr;
};

#endif //ANYSIM_SETTINGS_WIDGET_H
