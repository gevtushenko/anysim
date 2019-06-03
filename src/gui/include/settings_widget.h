//
// Created by egi on 6/3/19.
//

#ifndef ANYSIM_SETTINGS_WIDGET_H
#define ANYSIM_SETTINGS_WIDGET_H

#include <QWidget>
#include <QGroupBox>
#include <QLineEdit>

class settings_widget : public QWidget
{
Q_OBJECT

public:
  settings_widget ();

signals:
  void source_ready (double x, double y, double frequency);

private slots:
  void complete_source ();

private:
  QLineEdit *x_position;
  QLineEdit *y_position;
  QLineEdit *frequency;
  QGroupBox *source_box;
};

#endif //ANYSIM_SETTINGS_WIDGET_H
