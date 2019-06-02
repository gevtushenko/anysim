#include <iostream>
#include <memory>

#include "core/cpu/fdtd_2d.h"
#include "cpp/common_funcs.h"
#include "io/vtk.h"

#ifdef GUI_BUILD
#include "gui_simulation_manager.h"
#endif

int main (int argc, char *argv[])
{
  const double plane_size_x = 5;

  // const double dt = 1e-22;
  const double frequency = 2e+9;
  const double lambda_min = C0 / frequency;
  const double dx = lambda_min / 30;
  const auto optimal_nx = static_cast<unsigned int> (std::ceil (plane_size_x / dx));
  const auto optimal_ny = optimal_nx;
  const double plane_size_y = dx * optimal_ny;

  sources_holder<double> soft_source;

  // for (unsigned int j = 0; j < 3; j++)
  //   for (unsigned int i = 0; i < 3; i++)
  //     soft_source.append_source (frequency, ((j + 1) * optimal_ny/4) * optimal_nx + (i + 1) * optimal_nx / 4);
  soft_source.append_source (frequency, (optimal_ny/2) * optimal_nx + optimal_nx / 4);

  unsigned int slit_width = optimal_ny / 10;
  unsigned int slit_height = slit_width / 2;
  unsigned int distance_between_slits = slit_height * 4;

  unsigned int mid = optimal_ny / 2;
  unsigned int mid_rec_top = mid + distance_between_slits / 2;
  unsigned int mid_rec_bottom = mid - distance_between_slits / 2;
  unsigned int top_rec_bottom = mid_rec_top + slit_height;
  unsigned int bottom_rec_top = mid_rec_bottom - slit_height;

  unsigned int top_rec_height = optimal_ny - top_rec_bottom;
  unsigned int bottom_rec_height = bottom_rec_top;

  double er = 100.0;
  double mr = 100.0;

  rectangular_region_initializer top_rectangle (optimal_nx, optimal_ny, 2 * optimal_nx / 3, top_rec_bottom, slit_width, top_rec_height, er, mr);
  rectangular_region_initializer mid_rectangle (optimal_nx, optimal_ny, 2 * optimal_nx / 3, mid_rec_bottom, slit_width, distance_between_slits, er, mr);
  rectangular_region_initializer bot_rectangle (optimal_nx, optimal_ny, 2 * optimal_nx / 3, 0,              slit_width, bottom_rec_height, er, mr);

  fdtd_2d simulation (
      optimal_nx, optimal_ny,
      plane_size_x, plane_size_y,
      boundary_condition::periodic,
      boundary_condition::periodic,
      boundary_condition::periodic,
      boundary_condition::periodic,
      { &top_rectangle, &mid_rectangle, &bot_rectangle });
  // simulation.calculate (1000, soft_source);

#ifdef GPU_BUILD
  simulation.preprocess_gpu (soft_source);
#endif

  auto compute_function = [&simulation, &soft_source] (bool use_gpu)
  {
    simulation.calculate (10, soft_source, use_gpu);
  };

  auto render_function = [&simulation, &optimal_nx, &optimal_ny] (bool use_gpu, float *colors)
  {
    const auto coloring_begin = std::chrono::high_resolution_clock::now ();

#ifdef GPU_BUILD
    if (use_gpu)
    {
      fill_colors (optimal_nx, optimal_ny, simulation.get_d_ez (), colors);
    }
    else
#else
      cpp_unreferenced (use_gpu);
#endif
    {
      auto ez = simulation.get_ez ();
      for (unsigned int j = 0; j < optimal_ny; j++)
        for (unsigned int i = 0; i < optimal_nx; i++)
          for (unsigned int k = 0; k < 4; k++)
            fill_vertex_color (ez[j * optimal_nx + i], colors + 3 * 4 * (j * optimal_nx + i) + 3 * k);
    }
    const auto coloring_end = std::chrono::high_resolution_clock::now ();
    const std::chrono::duration<double> duration = coloring_end - coloring_begin;
    std::cout << "Coloring completed in " << duration.count () << "s\n";

  };

  gui_simulation_manager simulation_manager (
      argc, argv,
      optimal_nx, optimal_ny,
      plane_size_x, plane_size_y,
      compute_function, render_function);
  int ret_code = simulation_manager.run ();

#ifdef GPU_BUILD
  simulation.postprocess_gpu ();
#endif

  return ret_code;
}
