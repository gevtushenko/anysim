//
// Created by egi on 7/6/19.
//

#ifndef ANYSIM_SECTION_NAME_H
#define ANYSIM_SECTION_NAME_H

#include <QHBoxLayout>
#include <QLabel>
#include <QFrame>

class section_name : public QFrame
{
  Q_OBJECT

public:
  explicit section_name (const std::string &name);
};

#endif //ANYSIM_SECTION_NAME_H
