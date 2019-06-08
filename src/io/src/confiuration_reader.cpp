//
// Created by egi on 6/8/19.
//

#include "io/configuration_reader.h"
#include "io/json/json.hpp"

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

confituration_reader::~confituration_reader () = default;

confituration_reader::confituration_reader (const std::string &filename)
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

  if (configuration.find ("geometry") == configuration.end ())
  {
    std::cerr << "Can't get 'geometry' field from " << filename << std::endl;
    return;
  }

  geometry_width = configuration["geometry"]["width"];
  geometry_height = configuration["geometry"]["height"];
  cells_per_lambda = configuration["cells_per_lambda"];

  data = std::make_unique<json_wrapper> (configuration);
}

void confituration_reader::initialize_project (project_manager &pm)
{
  if (!is_valid ())
    return;

  pm.set_height (geometry_height);
  pm.set_width (geometry_width);
  pm.set_cells_per_lambda (cells_per_lambda);

  for (auto & [key, value] : data->json_content["sources"].items ())
  {
    std::cout << "Adding source #" << key << "\n";
    pm.append_source (value["frequency"], value["x"], value["y"]);
  }
}
