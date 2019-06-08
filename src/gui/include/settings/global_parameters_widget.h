//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_GLOBAL_PARAMETERS_WIDGET_H
#define ANYSIM_GLOBAL_PARAMETERS_WIDGET_H

#include <QWidget>

class QLineEdit;
class QGroupBox;

class global_parameters_widget : public QWidget
{
  Q_OBJECT
public:
  global_parameters_widget ();

  QLineEdit *cells_per_lambda = nullptr;
  QGroupBox *global_parameters_box = nullptr;

signals:
  void cells_per_lambda_changed (unsigned int);

private slots:
  void change_cells_per_lambda ();
};

#endif //ANYSIM_GLOBAL_PARAMETERS_WIDGET_H
