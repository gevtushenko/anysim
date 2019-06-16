//
// Created by egi on 6/15/19.
//

#include "core/config/configuration_node.h"

#include <iostream>

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

const std::vector<configuration_node>& configuration_node::group () const
{
  return children;
}

std::vector<configuration_node>& configuration_node::group (unsigned int group_id)
{
  return child (group_id).children;
}

const std::vector<configuration_node>& configuration_node::group (unsigned int group_id) const
{
  return child (group_id).children;
}

void configuration_node::print (unsigned int offset)
{
  std::string offset_str;

  for (unsigned int i = 0; i < offset; i++)
    offset_str += " ";

  if (type == configuration_node_type::void_value)
  {
    std::cout << offset_str << name << ":\n";

    for (auto &child: children)
      child.print (offset + 2);
  }
  else
  {
    if (type == configuration_node_type::int_value)
      std::cout << offset_str << name << " -> " << std::get<int> (value) << std::endl;
    if (type == configuration_node_type::double_value)
      std::cout << offset_str << name << " -> " << std::get<double> (value) << std::endl;
  }
}
