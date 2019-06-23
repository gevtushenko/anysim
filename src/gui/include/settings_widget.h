//
// Created by egi on 6/3/19.
//

#ifndef ANYSIM_SETTINGS_WIDGET_H
#define ANYSIM_SETTINGS_WIDGET_H

#include <QWidget>

class project_manager;
class QVBoxLayout;

class settings_widget : public QWidget
{
Q_OBJECT

public:
  settings_widget (project_manager &pm_arg);

public slots:
  void setup_configuration_node (std::size_t node_id);

private:
  project_manager &pm;
  QVBoxLayout *main_layout = nullptr;
  QVBoxLayout *node_layout = nullptr;
};

#endif //ANYSIM_SETTINGS_WIDGET_H
