//
// Created by egi on 6/1/19.
//

#ifndef ANYSIM_MODEL_WIDGET_H
#define ANYSIM_MODEL_WIDGET_H

#include <QWidget>
#include <QTreeView>

class model_widget : public QWidget
{
  Q_OBJECT

public:
  model_widget ();

private slots:
  void on_tree_view_context_menu (const QPoint &pos);

private:
  QTreeView *view;

};

#endif //ANYSIM_MODEL_WIDGET_H
