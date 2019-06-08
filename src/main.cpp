#include <iostream>
#include <memory>

#include "core/sm/con_simulation_manager.h"
#include "core/pm/project_manager.h"
#include "core/cpu/fdtd_2d.h"
#include "cpp/common_funcs.h"
#include "io/configuration_reader.h"

#ifndef CON_BUILD
#include "gui_simulation_manager.h"
#endif

simulation_manager *create_simulation_manager (bool console_run, project_manager &pm, int argc, char *argv[])
{
  cpp_unreferenced (console_run, argc, argv);

#ifndef CON_BUILD
  if (!console_run)
    return new gui_simulation_manager (pm, argc, argv);
#endif

  return new con_simulation_manager (pm);
}

int main (int argc, char *argv[])
{
  project_manager pm (false /* use double precision */);

  if (argc > 1)
  {
    confituration_reader config (argv[1]);
    config.initialize_project (pm);
  }

  std::unique_ptr<simulation_manager> simulation (create_simulation_manager (false, pm, argc, argv));
  return simulation->run ();
}
