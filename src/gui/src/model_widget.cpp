//
// Created by egi on 6/1/19.
//

#include "model_widget.h"
#include "tree_model.h"

#include <QStandardItemModel>
#include <QVBoxLayout>
#include <QLabel>
#include <QMenu>

model_widget::model_widget ()
{
  auto main_layout = new QVBoxLayout ();
  auto widget_label = new QLabel ("Model");

  QStringList headers;
  headers << "Test 1";

  // auto model = new TreeModel (headers, "123\n321");
  auto model = new QStandardItemModel ();
  auto root = model->invisibleRootItem ();
  auto item = new QStandardItem ("Project");
  item->setIcon (style ()->standardIcon (QStyle::SP_DirHomeIcon));
  root->appendRow (item);

  view = new QTreeView ();
  view->setHeaderHidden (true);
  view->setRootIsDecorated (false);
  view->setModel (model);
  view->setContextMenuPolicy (Qt::CustomContextMenu);
  connect (view, SIGNAL (customContextMenuRequested (QPoint)), this, SLOT (on_tree_view_context_menu (QPoint)));

  main_layout->addWidget (widget_label);
  main_layout->addWidget (view);

  setLayout (main_layout);
}

void model_widget::on_tree_view_context_menu (const QPoint &pos)
{
  auto menu = new QMenu (this);
  menu->addAction (QString ("Test"));
  menu->popup (view->viewport ()->mapToGlobal (pos));
}