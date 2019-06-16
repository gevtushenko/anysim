#include <iostream>
#include <memory>

#include "core/cpu/fdtd_2d.h"
#include "core/pm/project_manager.h"
#include "core/sm/simulation_manager.h"
#include "cpp/common_funcs.h"
#include "io/con/con_parser.h"

#include "core/cpu/euler_2d.h"

#ifndef CON_BUILD
#include "gui_application.h"
#endif

int start_application (bool console_run, project_manager &pm, int argc, char *argv[])
{
  cpp_unreferenced (console_run, argc, argv);

#ifndef CON_BUILD
  if (!console_run)
    return gui_application (pm, argc, argv).run ();
#endif

  return pm.run ();
}

int main (int argc, char *argv[])
{
  project_manager pm;

  con_parser args;
  if (args.parse (argc, argv, true /* require configuration */, pm))
    return 0;

  return start_application (false, pm, argc, argv);
}
