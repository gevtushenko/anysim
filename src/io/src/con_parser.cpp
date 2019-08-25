//
// Created by egi on 6/8/19.
//

#include "io/con/con_parser.h"
#include "io/configuration_reader.h"
#include "io/con/argagg/argagg.hpp"
#include "io/hdf5/hdf5_writer.h"
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
    { "configuration", {"-c", "--config"},   "load configuration file for simulation", 1 /* option arguments count */ },
    { "output",        {"-o", "--output"},   "dump results into file", 1 /* option arguments count */ }
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

  if (args["output"])
  {
    const auto output = args["output"].as<std::string> ();
    auto dumper = new hdf5_writer (output, pm);
    if (dumper->open ())
      pm.append_extractor_to_own (dumper);
  }

  if (args["configuration"])
  {
    configuration_reader config (args["configuration"].as<std::string> ());
    return config.initialize_project (pm);
  }
  else if (require_configuration)
  {
    std::cerr << "Usage: " << argv[0] << " --config=configuration_file.json" << std::endl;
    return true;
  }

  return false;
}
