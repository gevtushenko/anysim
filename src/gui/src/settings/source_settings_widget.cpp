//
// Created by egi on 6/8/19.
//

#include "settings/source_settings_widget.h"

#include <QVBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QGroupBox>
#include <QLabel>

source_settings_widget::source_settings_widget ()
{
  source_box = new QGroupBox ("Source");

  auto *vbox = new QVBoxLayout ();
  auto button = new QPushButton ("Create");
  x_position = new QLineEdit ("2.5");
  y_position = new QLineEdit ("2.5");
  frequency = new QLineEdit ("2E+9");
  vbox->addWidget (x_position);
  vbox->addWidget (y_position);
  vbox->addWidget (frequency);
  vbox->addWidget (button);

  connect (button, SIGNAL (clicked ()), this, SLOT (complete_source ()));

  source_box->setLayout (vbox);

  auto main_layout = new QVBoxLayout ();
  main_layout->addWidget (source_box);
  setLayout (main_layout);
}

void source_settings_widget::complete_source ()
{
  emit source_ready (
      x_position->text ().toDouble (),
      y_position->text ().toDouble (),
      frequency->text ().toDouble ());
}
