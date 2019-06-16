//
// Created by egi on 6/2/19.
//

#include "core/pm/project_manager.h"
#include "core/config/configuration.h"
#include "core/sm/simulation_manager.h"
#include "core/solver/solver.h"
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

  simulation->fill_configuration_scheme (*solver_configuration_scheme);
}

const configuration& project_manager::get_configuration_scheme () const
{
  return *solver_configuration_scheme;
}

configuration &project_manager::get_configuration ()
{
  return *solver_configuration;
}

int project_manager::run ()
{
  if (version != solver_configuration->get_version ())
  {
    version = solver_configuration->get_version ();
    simulation->apply_configuration (*solver_configuration);
  }

  simulation->calculate_next_time_step ();
  return 0;
}

const workspace &project_manager::get_solver_workspace () const
{
  return *solver_workspace;
}

simulation_manager &project_manager::get_simulation_manager ()
{
  return *simulation;
}
