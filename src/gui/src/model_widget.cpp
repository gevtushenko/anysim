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
  auto project = new QStandardItem ("Project");
  project->setIcon (style ()->standardIcon (QStyle::SP_DirHomeIcon));
  root->appendRow (project);

  auto global_definitions = new QStandardItem ("Global Definitions");
  auto global_definitions_parameters = new QStandardItem ("Parameters");
  auto global_definitions_materials = new QStandardItem ("Materials");
  auto sources = new QStandardItem ("Sources");

  project->appendRow (global_definitions);
  global_definitions->appendRow (global_definitions_parameters);
  global_definitions->appendRow (global_definitions_materials);
  project->appendRow (sources);

  view = new QTreeView ();
  view->setHeaderHidden (true);
  // view->setRootIsDecorated (false);
  view->setModel (model);
  view->setAnimated (true);
  view->setContextMenuPolicy (Qt::CustomContextMenu);
  view->expandAll ();
  connect (view, SIGNAL (customContextMenuRequested (QPoint)), this, SLOT (on_tree_view_context_menu (QPoint)));

  view->adjustSize ();

  main_layout->addWidget (widget_label);
  main_layout->addWidget (view);

  setLayout (main_layout);
}

void model_widget::on_tree_view_context_menu (const QPoint &pos)
{
  auto index = view->indexAt (pos);

  if (!index.isValid ())
    return;

  if (index.data ().toString () == "Sources")
    {
      auto menu = new QMenu (this);
      menu->addAction (QString ("Create source"), this, SLOT (create_source_slot ()));
      menu->popup (view->viewport ()->mapToGlobal (pos));
    }
}

void model_widget::create_source_slot ()
{
  emit create_source ();
}
