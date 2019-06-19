//
// Created by egi on 6/15/19.
//

#ifndef ANYSIM_CONFIGURATION_NODE_H
#define ANYSIM_CONFIGURATION_NODE_H

#include <vector>
#include <string>
#include <variant>

enum class configuration_node_type
{
  bool_value, int_value, double_value, string_value, void_value
};

template <class data_type>
configuration_node_type get_configuration_node_type ()
{
  static_assert (true, "Unsupported data type!");
  return configuration_node_type::void_value;
}

template <> inline configuration_node_type get_configuration_node_type<bool> () { return configuration_node_type::bool_value; }
template <> inline configuration_node_type get_configuration_node_type<int> () { return configuration_node_type::int_value; }
template <> inline configuration_node_type get_configuration_node_type<double> () { return configuration_node_type::double_value; }
template <> inline configuration_node_type get_configuration_node_type<std::string> () { return configuration_node_type::string_value; }

class configuration_node
{
public:
  using value_type = std::variant<bool, int, double, std::string>;

  configuration_node ();
  explicit configuration_node (const std::string &node_name);
  configuration_node (std::string node_name, configuration_node_type node_type, value_type node_value);

  template <class data_type>
  configuration_node &append_node (const std::string &node_name, const data_type &node_value)
  {
    update_version ();
    append (node_name, get_configuration_node_type<data_type> (), node_value);
    return *this;
  }

  configuration_node &append_group (const std::string &node_name)
  {
    update_version ();
    append (node_name);
    return *this;
  }

  configuration_node &append_and_get_group (const std::string &node_name)
  {
    update_version ();
    const size_t gid = children.size ();
    append (node_name);
    return children.at (gid);
  }

  std::size_t get_version () const;

  void update_version ()
  {
    version++;
  }

  void print (unsigned int offset=0);

  bool is_leaf () const { return children.empty (); }
  bool is_group () const { return type == configuration_node_type::void_value; }

  configuration_node &child (unsigned int child_id);
  const configuration_node &child (unsigned int child_id) const;

  std::vector<configuration_node> &group (unsigned int group_id);
  std::vector<configuration_node> &group ();
  const std::vector<configuration_node> &group () const;
  const std::vector<configuration_node> &group (unsigned int group_id) const;

private:
  void append (const std::string &node_name, configuration_node_type node_type, const value_type &node_value);
  void append (const std::string &node_name);

public:
  const std::string name;
  const configuration_node_type type;
  value_type value;

  static std::size_t version;
  std::vector<configuration_node> children;
};

#endif //ANYSIM_CONFIGURATION_NODE_H
