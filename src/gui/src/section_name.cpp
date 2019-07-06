//
// Created by egi on 7/6/19.
//

#include "section_name.h"

section_name::section_name (const std::string &name) : QFrame ()
{
  auto label = new QLabel (QString::fromStdString (name));
  auto layout = new QHBoxLayout ();
  layout->addWidget (label);
  setLayout (layout);
}

