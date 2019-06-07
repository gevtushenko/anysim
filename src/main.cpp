#include <iostream>
#include <memory>

#include "core/pm/project_manager.h"
#include "core/cpu/fdtd_2d.h"
#include "cpp/common_funcs.h"
#include "io/vtk.h"

#ifdef GUI_BUILD
#include "gui_simulation_manager.h"
#endif

#include "core/cpu/thread_pool.h"

int main (int argc, char *argv[])
{
  thread_pool pool;
  pool.execute ([&] (unsigned int thread_id, unsigned int total_threads) {
    for (unsigned int thr = 0; thr < total_threads; thr++)
    {
      if (thr == thread_id)
        std::cout << "Hello from " << thread_id << "/" << total_threads << std::endl;
      pool.barrier ();
    }
  });
  pool.execute ([&] (unsigned int thread_id, unsigned int total_threads) {
    for (unsigned int thr = 0; thr < total_threads; thr++)
    {
      if (thr == thread_id)
        std::cout << "Goodbuy from " << thread_id << "/" << total_threads << std::endl;
      pool.barrier ();
    }
  });

  gui_simulation_manager simulation_manager (argc, argv);
  int ret_code = simulation_manager.run ();
  return ret_code;
}
