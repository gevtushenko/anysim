#include <iostream>
#include <memory>

#include "core/pm/project_manager.h"
#include "core/cpu/fdtd_2d.h"
#include "cpp/common_funcs.h"
#include "io/vtk.h"

#ifdef GUI_BUILD
#include "gui_simulation_manager.h"
#endif

int main (int argc, char *argv[])
{
  // unsigned int slit_width = optimal_ny / 10;
  // unsigned int slit_height = slit_width / 2;
  // unsigned int distance_between_slits = slit_height * 4;

  // unsigned int mid = optimal_ny / 2;
  // unsigned int mid_rec_top = mid + distance_between_slits / 2;
  // unsigned int mid_rec_bottom = mid - distance_between_slits / 2;
  // unsigned int top_rec_bottom = mid_rec_top + slit_height;
  // unsigned int bottom_rec_top = mid_rec_bottom - slit_height;

  // unsigned int top_rec_height = optimal_ny - top_rec_bottom;
  // unsigned int bottom_rec_height = bottom_rec_top;

  // double er = 100.0;
  // double mr = 100.0;

  // rectangular_region_initializer top_rectangle (optimal_nx, optimal_ny, 2 * optimal_nx / 3, top_rec_bottom, slit_width, top_rec_height, er, mr);
  // rectangular_region_initializer mid_rectangle (optimal_nx, optimal_ny, 2 * optimal_nx / 3, mid_rec_bottom, slit_width, distance_between_slits, er, mr);
  // rectangular_region_initializer bot_rectangle (optimal_nx, optimal_ny, 2 * optimal_nx / 3, 0,              slit_width, bottom_rec_height, er, mr);

  // fdtd_2d simulation (
  //     optimal_nx, optimal_ny,
  //     plane_size_x, plane_size_y,
  //     boundary_condition::periodic,
  //     boundary_condition::periodic,
  //     boundary_condition::periodic,
  //     boundary_condition::periodic);
  // simulation.initialize_calculation_area (&top_rectangle);
  // simulation.initialize_calculation_area (&mid_rectangle);
  // simulation.initialize_calculation_area (&bot_rectangle);
  // simulation.calculate (1000, soft_source);

  gui_simulation_manager simulation_manager (argc, argv);
  int ret_code = simulation_manager.run ();

  return ret_code;
}
