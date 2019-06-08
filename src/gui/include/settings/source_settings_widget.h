//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_SOURCE_SETTINGS_WIDGET_H
#define ANYSIM_SOURCE_SETTINGS_WIDGET_H

#include <QWidget>

class QLineEdit;
class QGroupBox;

class source_settings_widget : public QWidget
{
  Q_OBJECT
public:
  source_settings_widget ();

  QLineEdit *x_position;
  QLineEdit *y_position;
  QLineEdit *frequency;
  QGroupBox *source_box;

signals:
  void source_ready (double x, double y, double frequency);

private slots:
  void complete_source ();
};

#endif //ANYSIM_SOURCE_SETTINGS_WIDGET_H
