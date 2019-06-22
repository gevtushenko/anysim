//
// Created by egi on 6/8/19.
//

#include "io/configuration_reader.h"
#include "io/json/json.hpp"

#include "core/config/configuration.h"
#include "core/pm/project_manager.h"

#include <iostream>
#include <fstream>

using json = nlohmann::json;

class json_wrapper
{
public:
  explicit json_wrapper (json &data) : json_content (std::move (data)) { }

  json json_content;
};

configuration_reader::~configuration_reader () = default;

configuration_reader::configuration_reader (const std::string &filename)
{
  std::ifstream file (filename);

  if (!file.is_open ())
  {
    std::cerr << "Can't open configuration file: " << filename << std::endl;
    return;
  }

  json configuration;
  file >> configuration;

  if (configuration.empty ())
  {
    std::cerr << "Configuration file " << filename << " is empty" << std::endl;
    return;
  }

  for (auto &required_option: {
    "max_time", "solver_name", "use_double_precision", "configuration", "project_name"
  })
  {
    if (configuration.find (required_option) == configuration.end ())
    {
      std::cerr << "Can't get required option: " << required_option << "!";
      return;
    }
  }

  max_time = configuration["max_time"];
  use_double_precision = configuration["use_double_precision"];
  solver_name = configuration["solver_name"];
  project_name = configuration["project_name"];

  data = std::make_unique<json_wrapper> (configuration["configuration"]);
}

static bool read_node (const configuration_node &scheme_node, configuration_node &config, const json &data)
{
  const auto &name = scheme_node.name;

  if (data.find (name) == data.end ())
  {
    std::cerr << "Error! Configuration file must contain '" << name << "' field." << std::endl;
    return true;
  }

  switch (scheme_node.type)
  {
    case configuration_node_type::bool_value:   config.append_node (name, data[name].get<bool> ());        break;
    case configuration_node_type::int_value:    config.append_node (name, data[name].get<int>  ());        break;
    case configuration_node_type::double_value: config.append_node (name, data[name].get<double> ());      break;
    case configuration_node_type::string_value: config.append_node (name, data[name].get<std::string> ()); break;
    case configuration_node_type::void_value:
      {
        auto group = config.append_and_get_group (name);
        for (auto &child: scheme_node.group ())
          if (read_node (*child, *group, data[name]))
            return true;
        break;
      }
    case configuration_node_type::array_type:
    {
      if (!data[name].is_array ())
      {
        std::cerr << "Error! Configuration field '" << name << " is supposed to be an array!" << std::endl;
        return true;
      }

      auto &array = config.append_and_get_array (name);
      auto &array_element_scheme = scheme_node.child (0);
      array.array_child_scheme = &array_element_scheme;

      unsigned int elem_id = 0;
      for (auto &elem: data[name])
      {
        auto &array_elem = array.append_and_get_array (std::to_string (elem_id++));
        for (auto &elem_field: array_element_scheme.group ())
          if (read_node (*elem_field, array_elem, elem))
            return true;
      }
      break;
    }
  }

  return false;
}

bool configuration_reader::initialize_project (project_manager &pm)
{
  if (!is_valid ())
    return true;

  pm.initialize (project_name, solver_name, use_double_precision);
  auto &scheme = pm.get_configuration_scheme ();
  auto &config = pm.get_configuration ();

  for (auto &scheme_node: scheme.get_root ().group ())
    if (read_node (*scheme_node, config.get_root (), data->json_content))
      return true;

  config.get_root ().print ();
  return false;
}
