//
// Created by egi on 6/15/19.
//

#include "core/config/configuration_node.h"

configuration_node::configuration_node (
    std::string node_name,
    configuration_node_type node_type,
    value_type node_value)
  : name (std::move (node_name))
  , type (node_type)
  , value (std::move (node_value))
{}

configuration_node::configuration_node (const std::string &node_name)
    : configuration_node (node_name, configuration_node_type::void_value, "group")
{}

configuration_node::configuration_node ()
  : configuration_node ("root")
{}

void configuration_node::append (const std::string &node_name)
{
  children.emplace_back (node_name);
}

void configuration_node::append (
    const std::string &node_name,
    configuration_node_type node_type,
    const configuration_node::value_type &node_value)
{
  children.emplace_back (node_name, node_type, node_value);
}

configuration_node& configuration_node::child (unsigned int child_id)
{
  return children.at (child_id);
}

const configuration_node& configuration_node::child (unsigned int child_id) const
{
  return children.at (child_id);
}

std::vector<configuration_node>& configuration_node::group (unsigned int group_id)
{
  return child (group_id).children;
}

const std::vector<configuration_node>& configuration_node::group (unsigned int group_id) const
{
  return child (group_id).children;
}
