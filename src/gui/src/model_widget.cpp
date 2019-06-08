//
// Created by egi on 6/1/19.
//

#include "model_widget.h"

#include <QStandardItemModel>
#include <QVBoxLayout>
#include <QLabel>
#include <QMenu>

#include "settings_widget.h"

model_widget::model_widget (settings_widget *settings_arg)
  : settings (settings_arg)
{
  auto main_layout = new QVBoxLayout ();
  auto widget_label = new QLabel ("Model");

  QStringList headers;
  headers << "Test 1";

  // auto model = new TreeModel (headers, "123\n321");
  auto model = new QStandardItemModel ();
  auto root = model->invisibleRootItem ();
  auto project = new QStandardItem (QIcon (":/icons/box.svg"), "Project");
  root->appendRow (project);
  project->setEditable (false);

  auto global_definitions = new QStandardItem (QIcon (":/icons/globe.svg"), "Global Definitions");
  auto global_definitions_parameters = new QStandardItem (QIcon (":/icons/sliders.svg"), "Parameters");
  auto global_definitions_materials = new QStandardItem (QIcon (":/icons/layers.svg"), "Materials");
  sources = new QStandardItem (QIcon (":/icons/radio.svg"), "Sources");

  global_definitions->setEditable (false);
  global_definitions_parameters->setEditable (false);
  global_definitions_materials->setEditable (false);
  sources->setEditable (false);

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
  connect (view, SIGNAL (clicked (const QModelIndex &)), this, SLOT (on_tree_view_clicked (const QModelIndex &)));

  connect (this, SIGNAL (update_global_parameters ()), settings, SLOT (show_global_parameters ()));
  connect (this, SIGNAL (create_source ()), settings, SLOT (show_source_settings ()));

  view->adjustSize ();

  main_layout->addWidget (widget_label);
  main_layout->addWidget (view);

  setLayout (main_layout);
}

#include <iostream>

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

void model_widget::on_tree_view_clicked (const QModelIndex &index)
{
  auto selected = index.data ().toString ().toStdString ();

  if (selected == "Parameters")
    emit update_global_parameters ();
}

void model_widget::create_source_slot ()
{
  sources->appendRow (new QStandardItem (QString ("Source %1").arg (last_source_id++)));
  emit create_source ();
}
