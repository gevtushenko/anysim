//
// Created by egi on 6/8/19.
//

#include "core/sm/con_simulation_manager.h"
#include "core/pm/project_manager.h"

con_simulation_manager::con_simulation_manager (project_manager &pm_arg)
  : pm (pm_arg)
{ }

int con_simulation_manager::run ()
{
  pm.prepare_simulation ();
  pm.calculate (1000);
  return 0;
}
