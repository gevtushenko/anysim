//
// Created by egi on 6/8/19.
//

#include "io/con/con_parser.h"
#include "io/configuration_reader.h"
#include "io/include/io/con/argagg/argagg.hpp"
#include "core/pm/project_manager.h"

#include <iostream>

class argagg_wrapper
{
public:
  argagg::parser parser;
};

con_parser::con_parser ()
  : parser_wrapper (new argagg_wrapper ())
{
  parser_wrapper->parser = {{
    { "help",          {"-h", "--help" },    "shows this help message", 0 /* option arguments count */},
    { "use_gpu",       {"-g", "--use-gpu" }, "allows simulation manager to use GPU", 0 /* option arguments count */},
    { "gpu_device",    {"-d", "--gpu-dev" }, "specify gpu device number", 1 /* option arguments count */},
    { "configuration", {"-c", "--config"},   "load configuration file for simulation", 1 /* option arguments count */ }
  }};
}

con_parser::~con_parser () = default;

bool con_parser::parse (int argc, char **argv, bool require_configuration, project_manager &pm)
{
  argagg::parser_results args;

  try
  {
    args = parser_wrapper->parser.parse (argc, argv);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what () << std::endl;
    return true;
  }

  if (args["help"])
  {
    std::cerr << parser_wrapper->parser;
    return true;
  }

  if (args["configuration"])
  {
    confituration_reader config (args["configuration"].as<std::string> ());
    config.initialize_project (pm);
  }
  else if (require_configuration)
  {
    std::cerr << "Usage: " << argv[0] << " --config=configuration_file.json" << std::endl;
    return true;
  }

  if (args["use_gpu"])
  {
    pm.set_use_gpu (args["gpu_device"].as<int> (0));
  }

  return false;
}
