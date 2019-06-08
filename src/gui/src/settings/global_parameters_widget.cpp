//
// Created by egi on 6/8/19.
//

#include "settings/global_parameters_widget.h"

#include <QVBoxLayout>
#include <QPushButton>
#include <QGroupBox>
#include <QLineEdit>

global_parameters_widget::global_parameters_widget ()
{
  auto *vbox = new QVBoxLayout ();
  auto button = new QPushButton ("Apply");
  cells_per_lambda = new QLineEdit ("40");
  vbox->addWidget (cells_per_lambda);
  vbox->addWidget (button);

  connect (button, SIGNAL (clicked ()), this, SLOT (change_cells_per_lambda ()));

  global_parameters_box = new QGroupBox ("Global parameters");
  global_parameters_box->setLayout (vbox);

  auto main_layout = new QVBoxLayout ();
  main_layout->addWidget (global_parameters_box);
  setLayout (main_layout);
}

void global_parameters_widget::change_cells_per_lambda ()
{
  emit cells_per_lambda_changed (cells_per_lambda->text ().toUInt ());
}