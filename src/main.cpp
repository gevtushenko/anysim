#include <iostream>
#include <memory>

#include "core/pm/project_manager.h"
#include "core/cpu/fdtd_2d.h"
#include "cpp/common_funcs.h"
#include "io/configuration_reader.h"

#ifdef GUI_BUILD
#include "gui_simulation_manager.h"
#endif

int main (int argc, char *argv[])
{
  project_manager pm (false /* use double precision */);

  if (argc > 1)
  {
    confituration_reader config (argv[1]);
    config.initialize_project (pm);
  }

  gui_simulation_manager simulation_manager (pm, argc, argv);
  int ret_code = simulation_manager.run ();
  return ret_code;
}
