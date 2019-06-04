//
// Created by egi on 6/1/19.
//

#ifndef ANYSIM_MODEL_WIDGET_H
#define ANYSIM_MODEL_WIDGET_H

#include <QWidget>
#include <QTreeView>

class QStandardItem;

class model_widget : public QWidget
{
  Q_OBJECT

public:
  model_widget ();

private slots:
  void on_tree_view_context_menu (const QPoint &pos);
  void create_source_slot ();

signals:
  void create_source ();

private:
  QTreeView *view = nullptr;
  QStandardItem *sources = nullptr;
  unsigned int last_source_id = 0;
};

#endif //ANYSIM_MODEL_WIDGET_H
