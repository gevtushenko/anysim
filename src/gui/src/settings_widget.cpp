//
// Created by egi on 6/3/19.
//

#include "settings_widget.h"
#include "settings/global_parameters_widget.h"
#include "settings/source_settings_widget.h"
#include "core/config/configuration.h"
#include "core/pm/project_manager.h"
#include "section_name.h"

#include <QPushButton>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QLabel>

settings_widget::settings_widget (project_manager &pm_arg)
  : QFrame ()
  , pm (pm_arg)
{
  main_layout = new QVBoxLayout ();
  node_layout = new QVBoxLayout ();
  auto widget_label = new section_name ("Settings");

  main_layout->addWidget (widget_label);
  main_layout->addLayout (node_layout);
  main_layout->addStretch (1);

  setLayout (main_layout);
}

static void clear_layout (QLayout *layout)
{
  if (!layout)
    return;

  while (auto item = layout->takeAt (0))
  {
    delete item->widget ();
    clear_layout (item->layout ());
  }
}


void settings_widget::setup_configuration_node (std::size_t parent_node_id)
{
  clear_layout (node_layout);
  auto &config = pm.get_configuration ();
  for (auto &node_id: config.children_for (parent_node_id))
  {
    if (config.is_group (node_id) || config.is_array (node_id))
      continue;

    auto param_layout = new QHBoxLayout ();
    auto param_name = new QLabel (QString::fromStdString (config.get_node_name (node_id)));
    auto param_value = new QLineEdit (QString::fromStdString (config.to_string (node_id)));

    param_layout->addWidget (param_name);
    param_layout->addWidget (param_value);

    connect (param_value, &QLineEdit::textChanged, this, [&,node_id] (const QString &new_value)
    {
      auto &config = pm.get_configuration ();
      auto type = config.get_node_type (node_id);
      if (type == int_type)
        config.update_value (node_id, new_value.toInt ());
      if (type == double_type)
        config.update_value (node_id, new_value.toDouble ());
      config.update_version ();
    });

    node_layout->addLayout (param_layout);
  }

  show ();
}

