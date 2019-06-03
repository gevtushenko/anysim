//
// Created by egi on 6/3/19.
//

#include "settings_widget.h"

#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>

settings_widget::settings_widget ()
{
  auto main_layout = new QVBoxLayout ();
  auto widget_label = new QLabel ("Settings");
  source_box = new QGroupBox ("Source");

  main_layout->addWidget (widget_label);
  main_layout->addWidget (source_box);
  main_layout->addStretch (1);

  auto *vbox = new QVBoxLayout ();
  auto button = new QPushButton ("Create");
  x_position = new QLineEdit ("2.5");
  y_position = new QLineEdit ("2.5");
  frequency = new QLineEdit ("2E+9");
  vbox->addWidget (x_position);
  vbox->addWidget (y_position);
  vbox->addWidget (frequency);
  vbox->addWidget (button);

  source_box->setLayout (vbox);

  connect (button, SIGNAL (clicked ()), this, SLOT (complete_source ()));

  setLayout (main_layout);
}

void settings_widget::complete_source ()
{
  emit source_ready (
      x_position->text ().toDouble (),
      y_position->text ().toDouble (),
      frequency->text ().toDouble ());
}
