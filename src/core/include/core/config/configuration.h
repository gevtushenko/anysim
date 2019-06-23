//
// Created by egi on 6/15/19.
//

#ifndef ANYSIM_CONFIGURATION_H
#define ANYSIM_CONFIGURATION_H

#include <type_traits>
#include <algorithm>
#include <vector>
#include <string>

enum configuration_value_type
{
  bool_type, int_type, double_type, string_type, array_type, group_type, unknown_type
};

template <class data_type>
configuration_value_type get_configuration_node_type ()
{
  static_assert (true, "Unsupported data type!");
  return configuration_value_type::unknown_type;
}

template <> inline configuration_value_type get_configuration_node_type<bool> () { return bool_type; }
template <> inline configuration_value_type get_configuration_node_type<int> () { return int_type; }
template <> inline configuration_value_type get_configuration_node_type<double> () { return double_type; }
template <> inline configuration_value_type get_configuration_node_type<std::string> () { return string_type; }

template<class T> struct dependent_false : std::false_type {};

struct configuration_value
{
  std::size_t data_id;

  int *int_storage;
  double *dbl_storage;
  std::string *str_storage;

  template <class T>
  operator T () const /// Implicitly convert into T
  {
    if constexpr (std::is_integral<T>::value)               return int_storage[data_id];
    else if constexpr (std::is_floating_point<T>::value)    return dbl_storage[data_id];
    else if constexpr (std::is_same<T, std::string>::value) return str_storage[data_id];
    else static_assert (dependent_false<T>::value, "Unsupported data type!");
    return {};
  }
};

/**
 * Configuration tree. It's supposed that accessing non constant
 * version of get_root updates configuration.
 *
 */
class configuration
{
public:
  configuration ();

  std::size_t get_root () const { return root; }

  std::size_t create_array (const std::string &array_name, std::size_t array_element_scheme_id)
  {
    return create_node (array_name, static_cast<int> (array_element_scheme_id), array_type);
  }

  std::size_t create_array (std::size_t parent_id, const std::string &array_name, std::size_t array_element_scheme_id)
  {
    const std::size_t array_id = create_array (array_name, array_element_scheme_id);
    add_child (parent_id, array_id);
    return array_id;
  }

  std::size_t create_group (const std::string &group_name)
  {
    const auto node_id = nodes_count++;
    std::size_t data_id = undefined_data_id;
    nodes_types.emplace_back (group_type);
    nodes_names.emplace_back (group_name);
    nodes_data_id.emplace_back (data_id);
    nodes_children.emplace_back ();
    return node_id;
  }

  std::size_t create_group (std::size_t parent_id, const std::string &group_name)
  {
    const std::size_t group_id = create_group (group_name);
    add_child (parent_id, group_id);
    return group_id;
  }

  template <class data_type>
  std::size_t create_node (const std::string &node_name, const data_type &value) { return create_node (node_name, value, get_configuration_node_type<data_type> ()); }

  template <class data_type>
  std::size_t create_node (std::size_t parent_id, const std::string &node_name, const data_type &value)
  {
    const std::size_t node_id = create_node (node_name, value, get_configuration_node_type<data_type> ());
    add_child (parent_id, node_id);
    return node_id;
  }

  configuration_value_type get_node_type (std::size_t node_id) const
  {
    return nodes_types[node_id];
  }

  configuration_value get_node_value (std::size_t node_id) const
  {
    return {
      nodes_data_id[node_id],
      int_storage.data (),
      dbl_storage.data (),
      str_storage.data ()
    };
  }

  std::size_t clone_node (std::size_t node_id, const configuration *config=nullptr)
  {
    const auto node_type = config ? config->get_node_type (node_id) : get_node_type (node_id);
    const auto node_data_id = config ? config->nodes_data_id[node_id] : nodes_data_id[node_id];
    const auto clone_id = nodes_count++;
    std::size_t data_id = undefined_data_id;

    if (node_type != group_type)
    {
      if (node_type == string_type)
      {
        data_id = str_storage.size ();

        if (config)
          str_storage.push_back (config->str_storage[node_data_id]);
        else
          str_storage.push_back (str_storage[node_data_id]);
      }
      else if (node_type == double_type)
      {
        data_id = dbl_storage.size ();

        if (config)
          dbl_storage.push_back (config->dbl_storage[node_data_id]);
        else
          dbl_storage.push_back (dbl_storage[node_data_id]);
      }
      else
      {
        data_id = int_storage.size ();

        if (config)
          int_storage.push_back (config->int_storage[node_data_id]);
        else
          int_storage.push_back (int_storage[node_data_id]);
      }
    }

    nodes_types.emplace_back (node_type);
    nodes_names.push_back (config ? config->get_node_name (node_id) : nodes_names[node_id]);
    nodes_data_id.push_back (data_id);
    nodes_children.emplace_back ();

    for (auto &child: config ? config->children_for (node_id) : children_for (node_id))
      add_child (clone_id, clone_node (child, config));

    return clone_id;
  }

  void update_version () { version++; }
  std::size_t get_version () const { return version; }

  template <class data_type>
  void update_value (std::size_t node_id, data_type new_value)
  {
    get_storage<data_type> ()->at (nodes_data_id[node_id]) = new_value;
  }

  std::string to_string (std::size_t node_id)
  {
    const auto node_data_id = nodes_data_id[node_id];
    switch (get_node_type (node_id))
    {
      case string_type: return str_storage[node_data_id];
      case int_type:    return std::to_string (int_storage[node_data_id]);
      case double_type: return std::to_string (dbl_storage[node_data_id]);
      default:          return { };
    }

    return { };
  }

  bool is_group (std::size_t node_id) const { return get_node_type (node_id) == group_type; }
  bool is_array (std::size_t node_id) const { return get_node_type (node_id) == array_type; }

  const std::string get_node_name (std::size_t node_id) const { return nodes_names[node_id]; }

  void add_child (std::size_t parent, std::size_t child)
  {
    auto &children = nodes_children[parent];
    if (std::find (children.begin (), children.end (), child) == children.end ())
      children.push_back (child);
  }

  void add_children (std::size_t parent, const std::vector<std::size_t> &children)
  {
    for (auto &child: children)
      add_child (parent, child);
  }

  const std::vector<std::size_t> children_for (std::size_t parent) const { return nodes_children[parent]; }

private:
  template <class data_type>
  std::vector<data_type> *get_storage () { static_assert (true, "Unsupported data type!"); return nullptr; }

  template <class data_type>
  std::size_t create_node (const std::string &node_name, const data_type &value, configuration_value_type type);

private:
  std::size_t root = 0;
  std::size_t version = 0;
  std::size_t nodes_count = 0;
  mutable std::vector<int> int_storage;
  mutable std::vector<double> dbl_storage;
  mutable std::vector<std::string> str_storage;
  std::vector<std::string> nodes_names;
  std::vector<std::size_t> nodes_data_id;
  std::vector<configuration_value_type> nodes_types;
  std::vector<std::vector<std::size_t>> nodes_children;

  constexpr static const std::size_t undefined_data_id = std::numeric_limits<std::size_t>::max () - 1;
};

#endif //ANYSIM_CONFIGURATION_H
