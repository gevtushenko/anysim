//
// Created by egi on 6/15/19.
//

#ifndef ANYSIM_CONFIGURATION_H
#define ANYSIM_CONFIGURATION_H

#include <type_traits>
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
  configuration_value_type type;

  int *int_storage;
  double *dbl_storage;
  std::string *str_storage;

  template <class T>
  operator T () const /// Implicitly convert into T
  {
    if constexpr (std::is_same<T, int>::value)              return int_storage[data_id];
    else if constexpr (std::is_same<T, bool>::value)        return int_storage[data_id];
    else if constexpr (std::is_same<T, double>::value)      return dbl_storage[data_id];
    else if constexpr (std::is_same<T, std::string>::value) return str_storage[data_id];
    else
    {
      static_assert (dependent_false<T>::value, "Unsupported data type!");
      return {};
    }
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

  std::size_t create_node (const std::string &node_name)
  {
    const auto node_id = nodes_count++;
    nodes_names.emplace_back (node_name);
    nodes_data_id.emplace_back (undefined_data_id);
    nodes_children.emplace_back ();

    return node_id;
  }

  std::size_t create_node (const std::string &node_name, bool value)
  {
    return create_node (node_name, static_cast<int> (value));
  }

  template <class data_type>
  std::size_t create_node (
      const std::string &node_name,
      const data_type &value)
  {
    const auto node_id = nodes_count++;
    std::size_t data_id = undefined_data_id;
    configuration_value_type type = get_configuration_node_type<data_type> ();
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

  configuration_value get_node_value (std::size_t node_id) const
  {
    return {
      nodes_data_id[node_id],
      nodes_types[node_id],
      int_storage.data (),
      dbl_storage.data (),
      str_storage.data ()
    };
  }

private:
  template <class data_type>
  std::vector<data_type> *get_storage () { static_assert (true, "Unsupported data type!"); return nullptr; }

private:
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

template <> inline std::vector<int>         *configuration::get_storage<int> ()         { return &int_storage; }
template <> inline std::vector<double>      *configuration::get_storage<double> ()      { return &dbl_storage; }
template <> inline std::vector<std::string> *configuration::get_storage<std::string> () { return &str_storage; }

#endif //ANYSIM_CONFIGURATION_H
