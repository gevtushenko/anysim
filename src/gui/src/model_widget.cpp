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

static void append_to_model (const configuration &config, std::size_t parent_id, QStandardItem *parent)
{
  for (auto &node_id: config.children_for (parent_id))
  {
    if (config.is_group (node_id) || config.is_array (node_id))
    {
      auto new_item = new QStandardItem (QString::fromStdString (config.get_node_name (node_id)));
      new_item->setData (static_cast<unsigned int> (node_id), Qt::UserRole + 1);
      new_item->setEditable (false);
      parent->appendRow (new_item);
      append_to_model (config, node_id, new_item);
    }
  }
}

model_widget::model_widget (project_manager &pm_arg)
  : pm (pm_arg)
{
  auto main_layout = new QVBoxLayout ();
  auto widget_label = new QLabel ("Model");

  model = new QStandardItemModel ();
  auto root = model->invisibleRootItem ();
  auto project = new QStandardItem (QIcon (":/icons/box.svg"), QString::fromStdString (pm.get_project_name ()));
  root->appendRow (project);
  project->setEditable (false);

  const auto &config = pm.get_configuration ();
  append_to_model (config, config.get_root (), project);

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

  auto id = index.data (Qt::UserRole + 1).toUInt ();
  auto &config = pm.get_configuration ();

  if (config.is_array (id))
    {
      auto menu = new QMenu (this);
      auto parent = model->itemFromIndex (index);
      menu->addAction (QString ("Append element"), this, [=] () {
        auto &config = pm.get_configuration ();
        const int element_scheme_id = config.get_node_value (id);
        const auto clone_id = config.clone_node (element_scheme_id);
        auto new_item = new QStandardItem (QString::fromStdString (config.get_node_name (clone_id)));
        new_item->setData (static_cast<unsigned int> (clone_id), Qt::UserRole + 1);
        new_item->setEditable (false);
        parent->appendRow (new_item);
        config.add_child (id, clone_id);
        config.update_version ();

        append_to_model (config, clone_id, new_item);
      });
      menu->popup (view->viewport ()->mapToGlobal (pos));
    }
}

void model_widget::on_tree_view_clicked (const QModelIndex &index)
{
  auto id = index.data (Qt::UserRole + 1).toUInt ();
  emit configuration_node_selected (id);
}
