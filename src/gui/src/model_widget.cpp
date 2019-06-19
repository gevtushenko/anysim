//
// Created by egi on 6/1/19.
//

#include "model_widget.h"
#include "core/pm/project_manager.h"
#include "core/config/configuration.h"

#include <QStandardItemModel>
#include <QVBoxLayout>
#include <QLabel>
#include <QMenu>

#include "settings_widget.h"

static void append_to_model (const configuration_node &root, QStandardItem *parent, std::vector<configuration_node> &linearized_tree)
{
  for (const auto &node: root.group ())
  {
    if (node.is_group ())
    {
      auto new_item = new QStandardItem (QString::fromStdString (node.name));
      new_item->setData (static_cast<unsigned int> (linearized_tree.size ()), Qt::UserRole + 1);
      parent->appendRow (new_item);
      linearized_tree.push_back (node);
      append_to_model (node, new_item, linearized_tree);
    }
  }
}

model_widget::model_widget (project_manager &pm)
{
  auto main_layout = new QVBoxLayout ();
  auto widget_label = new QLabel ("Model");

  auto model = new QStandardItemModel ();
  auto root = model->invisibleRootItem ();
  auto project = new QStandardItem (QIcon (":/icons/box.svg"), QString::fromStdString (pm.get_project_name ()));
  root->appendRow (project);
  project->setEditable (false);

  append_to_model (pm.get_configuration ().get_root (), project, linearized_tree);

  // auto global_definitions = new QStandardItem (QIcon (":/icons/globe.svg"), "Global Definitions");
  // auto global_definitions_parameters = new QStandardItem (QIcon (":/icons/sliders.svg"), "Parameters");
  // auto global_definitions_materials = new QStandardItem (QIcon (":/icons/layers.svg"), "Materials");
  // sources = new QStandardItem (QIcon (":/icons/radio.svg"), "Sources");

  // global_definitions->setEditable (false);
  // global_definitions_parameters->setEditable (false);
  // global_definitions_materials->setEditable (false);
  // sources->setEditable (false);

  // project->appendRow (global_definitions);
  // global_definitions->appendRow (global_definitions_parameters);
  // global_definitions->appendRow (global_definitions_materials);
  // project->appendRow (sources);

  view = new QTreeView ();
  view->setHeaderHidden (true);
  // view->setRootIsDecorated (false);
  view->setModel (model);
  view->setAnimated (true);
  view->setContextMenuPolicy (Qt::CustomContextMenu);
  view->expandAll ();
  view->adjustSize ();

  connect (view, SIGNAL (customContextMenuRequested (QPoint)), this, SLOT (on_tree_view_context_menu (QPoint)));
  connect (view, SIGNAL (clicked (const QModelIndex &)), this, SLOT (on_tree_view_clicked (const QModelIndex &)));

  // connect (this, SIGNAL (update_global_parameters ()), settings, SLOT (show_global_parameters ()));
  // connect (this, SIGNAL (create_source ()), settings, SLOT (show_source_settings ()));

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
  auto id = index.data (Qt::UserRole + 1).toUInt ();
  std::cout << linearized_tree[id].name << std::endl;

  if (selected == "Parameters")
    emit update_global_parameters ();
}

void model_widget::create_source_slot ()
{
  sources->appendRow (new QStandardItem (QString ("Source %1").arg (last_source_id++)));
  emit create_source ();
}
