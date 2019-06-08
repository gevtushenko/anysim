#include <iostream>
#include <memory>

#include "core/pm/project_manager.h"
#include "core/cpu/fdtd_2d.h"
#include "cpp/common_funcs.h"

#ifdef GUI_BUILD
#include "gui_simulation_manager.h"
#endif

#include "core/cpu/thread_pool.h"

int main (int argc, char *argv[])
{
  gui_simulation_manager simulation_manager (argc, argv);
  int ret_code = simulation_manager.run ();
  return ret_code;
}
