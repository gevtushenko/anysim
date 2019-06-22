//
// Created by egi on 6/3/19.
//

#include "settings_widget.h"
#include "settings/global_parameters_widget.h"
#include "settings/source_settings_widget.h"

#include <QPushButton>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QLabel>

settings_widget::settings_widget ()
{
  main_layout = new QVBoxLayout ();
  node_layout = new QVBoxLayout ();
  auto widget_label = new QLabel ("Settings");

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

static std::string get_node_value (configuration_node &node)
{
  if (node.type == configuration_node_type::int_value)
    return std::to_string (std::get<int> (node.value));
  if (node.type == configuration_node_type::double_value)
    return std::to_string (std::get<double> (node.value));
  return "";
}

void settings_widget::setup_configuration_node (configuration_node *root)
{
  clear_layout (node_layout);
  for (auto &node: root->group ())
  {
    if (node->is_group () || node->is_array ())
      continue;

    auto param_layout = new QHBoxLayout ();
    auto param_name = new QLabel (QString::fromStdString (node->name));
    auto param_value = new QLineEdit (QString::fromStdString (get_node_value (*node)));

    param_layout->addWidget (param_name);
    param_layout->addWidget (param_value);

    connect (param_value, &QLineEdit::textChanged, this, [&] (const QString &new_value)
    {
      if (node->type == configuration_node_type::int_value)
        node->value = new_value.toInt ();
      if (node->type == configuration_node_type::double_value)
        node->value = new_value.toDouble ();
      node->update_version ();
    });

    node_layout->addLayout (param_layout);
  }

  show ();
}

