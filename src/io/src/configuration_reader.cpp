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

  if (configuration.find ("grid_initializer") != configuration.end ())
  {
    auto get_dir_name = [](const std::string &file) {
      std::size_t found = file.find_last_of ("/\\");
      return file.substr (0, found);
    };

    const std::string script_filename = configuration["grid_initializer"];
    const auto script_filepath = get_dir_name (filename) + "/" + script_filename;
    std::ifstream initializer (script_filepath);

    if (initializer.is_open ())
    {
      std::string file_content ((std::istreambuf_iterator<char>(initializer)), std::istreambuf_iterator<char>());
      initializer_script = std::move (file_content);
    }
    else
    {
      std::cerr << "Can't read initializer script from " << script_filepath << std::endl;
    }
  }

  max_time = configuration["max_time"];
  use_double_precision = configuration["use_double_precision"];
  solver_name = configuration["solver_name"];
  project_name = configuration["project_name"];

  data = std::make_unique<json_wrapper> (configuration["configuration"]);
}

static bool read_node (const configuration &scheme, std::size_t scheme_id, configuration &config, std::size_t config_id, const json &data)
{
  const auto name = scheme.get_node_name (scheme_id);

  if (data.find (name) == data.end ())
  {
    std::cerr << "Error! Configuration file must contain '" << name << "' field." << std::endl;
    return true;
  }

  switch (scheme.get_node_type (scheme_id))
  {
    case bool_type:   config.create_node (config_id, name, data[name].get<bool> ());        break;
    case int_type:    config.create_node (config_id, name, data[name].get<int>  ());        break;
    case double_type: config.create_node (config_id, name, data[name].get<double> ());      break;
    case string_type: config.create_node (config_id, name, data[name].get<std::string> ()); break;
    case group_type:
      {
        auto group_id = config.create_group (config_id, name);
        for (auto &scheme_child_id: scheme.children_for (scheme_id))
          if (read_node (scheme, scheme_child_id, config, group_id, data[name]))
            return true;
        break;
      }
    case array_type:
    {
      if (!data[name].is_array ())
      {
        std::cerr << "Error! Configuration field '" << name << " is supposed to be an array!" << std::endl;
        return true;
      }

      const int scheme_array_element_scheme_id = scheme.get_node_value (scheme_id);
      const int array_element_scheme_id = config.clone_node (scheme_array_element_scheme_id, &scheme);
      auto array_id = config.create_array (config_id, name, array_element_scheme_id);

      unsigned int elem_id = 0;
      for (auto &elem: data[name])
      {
        auto array_elem_id = config.create_group (array_id, std::to_string (elem_id++));
        for (auto elem_field_id: scheme.children_for (scheme_array_element_scheme_id))
          if (read_node (scheme, elem_field_id, config, array_elem_id, elem))
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

  pm.initialize (project_name, solver_name, max_time, use_double_precision);

  if (!initializer_script.empty ())
    pm.set_initializer_script (initializer_script);

  auto &scheme = pm.get_configuration_scheme ();
  auto &config = pm.get_configuration ();

  for (auto scheme_node_id: scheme.children_for (scheme.get_root ()))
    if (read_node (scheme, scheme_node_id, config, config.get_root (), data->json_content))
      return true;

  config.update_version ();
  return false;
}
