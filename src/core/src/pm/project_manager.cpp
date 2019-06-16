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
    bool use_double_precision_arg)
{
  solver_name = std::move (solver_arg);
  project_name = std::move (project_name_arg);
  use_double_precision = use_double_precision_arg;

  solver_configuration = std::make_unique<configuration> ();
  solver_configuration_scheme = std::make_unique<configuration> ();
  simulation = std::make_unique<simulation_manager> (solver_name, use_double_precision, *solver_workspace);

  auto &grid_part = solver_configuration_scheme->get_root ().append_and_get_group ("grid");
  grid_part.append_node ("nx", 10 /* default value */);
  grid_part.append_node ("ny", 10 /* default value */);
  grid_part.append_node ("width",  1.0 /* default value */);
  grid_part.append_node ("height", 1.0 /* default value */);

  auto &solver_part = solver_configuration_scheme->get_root ().append_and_get_group ("solver");
  simulation->fill_configuration_scheme (solver_part);
  solver_configuration_scheme->get_root ().print ();
}

const configuration& project_manager::get_configuration_scheme () const
{
  return *solver_configuration_scheme;
}

configuration &project_manager::get_configuration ()
{
  return *solver_configuration;
}

bool project_manager::run ()
{
  const auto &config = *solver_configuration;
  if (version != config.get_version ())
  {
    version = config.get_version ();

    auto &grid_node = config.get_root ().child (0);
    const unsigned int nx = std::get<int> (grid_node.child (0).value);
    const unsigned int ny = std::get<int> (grid_node.child (1).value);

    const double width  = std::get<double> (grid_node.child (2).value);
    const double height = std::get<double> (grid_node.child (3).value);

    solver_grid = std::make_unique<grid> (*solver_workspace, nx, ny, width, height);
    simulation->apply_configuration (config.get_root ().child (1), *solver_grid);
  }

  return simulation->calculate_next_time_step ();
}

const workspace &project_manager::get_solver_workspace () const
{
  return *solver_workspace;
}

simulation_manager &project_manager::get_simulation_manager ()
{
  return *simulation;
}

const grid &project_manager::get_grid () const
{
  return *solver_grid;
}
