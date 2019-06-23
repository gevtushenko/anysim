//
// Created by egi on 6/2/19.
//

#include "core/pm/project_manager.h"
#include "core/config/configuration.h"
#include "core/sm/simulation_manager.h"
#include "core/solver/solver.h"
#include "core/grid/grid.h"
#include "core/cpu/euler_2d.h"
#include "core/solver/workspace.h"

project_manager::project_manager ()
  : solver_workspace (new workspace ())
{ }

project_manager::~project_manager () = default;

void project_manager::initialize (
    std::string project_name_arg,
    std::string solver_arg,
    double max_simulation_time_arg,
    bool use_double_precision_arg)
{
  solver_name = std::move (solver_arg);
  project_name = std::move (project_name_arg);
  use_double_precision = use_double_precision_arg;

  solver_configuration = std::make_unique<configuration> ();
  solver_configuration_scheme = std::make_unique<configuration> ();
  simulation = std::make_unique<simulation_manager> (
      solver_name,
      max_simulation_time_arg,
      use_double_precision,
      *solver_workspace);

  auto &scheme = *solver_configuration_scheme;
  const auto grid_id = scheme.create_group (scheme.get_root (), "grid");
  scheme.create_node (grid_id, "nx", 10 /* default value */);
  scheme.create_node (grid_id, "ny", 10 /* default value */);
  scheme.create_node (grid_id, "width",  1.0 /* default value */);
  scheme.create_node (grid_id, "height", 1.0 /* default value */);
  const auto solver_id = scheme.create_group (scheme.get_root (), "solver");
  simulation->fill_configuration_scheme (scheme, solver_id);

  // configuration_node grid_part ("grid");
  // grid_part.append_node ("nx", 10 /* default value */);
  // grid_part.append_node ("ny", 10 /* default value */);
  // grid_part.append_node ("width",  1.0 /* default value */);
  // grid_part.append_node ("height", 1.0 /* default value */);
  // solver_configuration_scheme->get_root ().append_node (grid_part);

  // auto solver_part = solver_configuration_scheme->get_root ().append_and_get_group ("solver");
  // simulation->fill_configuration_scheme (*solver_part);
  // solver_configuration_scheme->get_root ().print ();
}

const configuration& project_manager::get_configuration_scheme () const
{
  return *solver_configuration_scheme;
}

configuration &project_manager::get_configuration ()
{
  return *solver_configuration;
}

void project_manager::update_project ()
{
  const auto &config = *solver_configuration;
  if (version != config.get_version ())
  {
    version = config.get_version ();

    auto grid_node_id = config.children_for (config.get_root ()).front ();
    auto grid_params = config.children_for (grid_node_id);
    const unsigned int nx = config.get_node_value (grid_params[0]);
    const unsigned int ny = config.get_node_value (grid_params[1]);

    const double width  = config.get_node_value (grid_params[2]);
    const double height = config.get_node_value (grid_params[3]);

    solver_grid = std::make_unique<grid> (*solver_workspace, nx, ny, width, height);
    simulation->apply_configuration (config, config.children_for (config.get_root ()).at (1), *solver_grid, gpu_num);
  }
}

bool project_manager::run ()
{
  update_project ();
  return simulation->calculate_next_time_step (extractors.data (), extractors.size ());
}

const workspace &project_manager::get_solver_workspace () const
{
  return *solver_workspace;
}

const grid &project_manager::get_grid () const
{
  return *solver_grid;
}

void project_manager::append_extractor (result_extractor *extractor)
{
  extractors.push_back (extractor);
}
