//
// Created by egi on 6/15/19.
//

#include "core/config/configuration.h"

// TODO Use momento pattern to easily undo configuration changes

configuration::configuration () = default;

template <class data_type>
std::size_t configuration::create_node (
    const std::string &node_name,
    const data_type &value,
    configuration_value_type type)
{
  const auto node_id = nodes_count++;
  std::size_t data_id = undefined_data_id;
  if (type != array_type && type != group_type)
  {
    auto storage = get_storage<data_type> ();
    data_id = storage->size ();
    storage->push_back (value);
  }

  nodes_types.emplace_back (type);
  nodes_names.emplace_back (node_name);
  nodes_data_id.emplace_back (data_id);
  nodes_children.emplace_back ();

  return node_id;
}

template <> std::vector<int>         *configuration::get_storage<int> ()         { return &int_storage; }
template <> std::vector<double>      *configuration::get_storage<double> ()      { return &dbl_storage; }
template <> std::vector<std::string> *configuration::get_storage<std::string> () { return &str_storage; }

template std::size_t configuration::create_node<int>         (const std::string &node_name, const int &);
template std::size_t configuration::create_node<double>      (const std::string &node_name, const double &);
template std::size_t configuration::create_node<std::string> (const std::string &node_name, const std::string &);
