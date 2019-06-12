#include <iostream>
#include <memory>

#include "core/sm/con_simulation_manager.h"
#include "core/pm/project_manager.h"
#include "core/cpu/fdtd_2d.h"
#include "cpp/common_funcs.h"
#include "io/con/con_parser.h"

#include "core/cpu/euler_2d.h"

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
  thread_pool threads;
  euler_2d<double> solver (threads, 400, 150);

  solver.calculate (10000);

  return 0;
  project_manager pm (false /* use double precision */);
  std::unique_ptr<simulation_manager> simulation (create_simulation_manager (false, pm, argc, argv));

  con_parser args;
  if (args.parse (argc, argv, simulation->require_configuration (), pm))
    return 0;

  return simulation->run ();
}
