#include <iostream>
#include <chrono>
#include <memory>
#include <cmath>

#include "core/cpu/sources_holder.h"
#include "core/gpu/fdtd_gpu_interface.h"
#include "io/vtk.h"

// TODO Extract
#include <cuda_runtime.h>

constexpr double C0 = 299792458; /// Speed of light [metres per second]

enum class boundary_condition
{
  dirichlet, periodic
};

template <class float_type>
class region_initializer
{
public:
  region_initializer () = delete;
  region_initializer (unsigned int nx_arg, unsigned int ny_arg)
    : nx (nx_arg)
    , ny (ny_arg)
  { }

  virtual void fill_region (float_type *er, float_type *hr) const = 0;

protected:
  const unsigned int nx, ny;
};

template <class float_type>
class rectangular_region_initializer : public region_initializer<float_type>
{
public:
  rectangular_region_initializer () = delete;
  rectangular_region_initializer (
      unsigned int nx,
      unsigned int ny,
      unsigned int region_i_start_arg,
      unsigned int region_j_start_arg,
      unsigned int width_arg,
      unsigned int height_arg,
      float_type er_value_arg,
      float_type hr_value_arg)
    : region_initializer<float_type> (nx, ny)
    , region_i_start (region_i_start_arg)
    , region_j_start (region_j_start_arg)
    , width (width_arg)
    , height (height_arg)
    , er_value (er_value_arg)
    , hr_value (hr_value_arg)
  { }

  void fill_region (float_type *er, float_type *hr) const final
  {
    using ri = region_initializer<float_type>;

    for (unsigned int j = region_j_start; j < region_j_start + height; j++)
    {
      for (unsigned int i = region_i_start; i < region_i_start + width; i++)
      {
        er[j * ri::nx + i] = er_value;
        hr[j * ri::nx + i] = hr_value;
      }
    }
  }

protected:
  const unsigned int region_i_start, region_j_start, width, height;
  const float_type er_value, hr_value;
};

template <class float_type>
class fdtd_2d
{
  boundary_condition left_bc, bottom_bc, right_bc, top_bc;

  const unsigned int nx, ny;
  const float_type dx, dy;
  const float_type dt;

  std::unique_ptr<float_type[]> m_e, m_h;  /// Compute update coefficients
  std::unique_ptr<float_type[]> cex, cey;  /// Curl E components
  std::unique_ptr<float_type[]> chz;       /// Curl H components
  std::unique_ptr<float_type[]> dz;
  std::unique_ptr<float_type[]> ez, hx, hy;
  std::unique_ptr<float_type[]> er, hr; /// Materials properties (for now assume mu_xx = mu_yy)

public:
  fdtd_2d () = delete;
  fdtd_2d (
      unsigned int nx_arg,
      unsigned int ny_arg,
      float_type plane_size_x,
      float_type plane_size_y,
      boundary_condition left_boundary_condition,
      boundary_condition bottom_boundary_condition,
      boundary_condition right_boundary_condition,
      boundary_condition top_boundary_condition,
      const std::vector<region_initializer<float_type>*> &initializers)
      : left_bc (left_boundary_condition)
      , bottom_bc (bottom_boundary_condition)
      , right_bc (right_boundary_condition)
      , top_bc (top_boundary_condition)
      , nx (nx_arg)
      , ny (ny_arg)
      , dx (plane_size_x / nx)
      , dy (plane_size_y / ny)
      , dt (std::min (dx, dy) / C0 / 2.0)
      , m_e (new float_type[nx * ny])
      , m_h (new float_type[nx * ny])
      , cex (new float_type[nx * ny])
      , cey (new float_type[nx * ny])
      , chz (new float_type[nx * ny])
      , dz  (new float_type[nx * ny])
      , ez  (new float_type[nx * ny])
      , hx  (new float_type[nx * ny])
      , hy  (new float_type[nx * ny])
      , er  (new float_type[nx * ny])
      , hr  (new float_type[nx * ny])
  {
    /// Assume that we are in free space
    std::fill_n (er.get (), nx * ny, 1.0);
    std::fill_n (hr.get (), nx * ny, 1.0);

    std::fill_n (hx.get (), nx * ny, 0.0);
    std::fill_n (hy.get (), nx * ny, 0.0);
    std::fill_n (ez.get (), nx * ny, 0.0);
    std::fill_n (dz.get (), nx * ny, 0.0);

    for (auto &initializer: initializers)
      initializer->fill_region (er.get (), hr.get ());

    for (unsigned int i = 0; i < nx * ny; i++)
      m_e[i] = C0 * dt / er[i];
    for (unsigned int i = 0; i < nx * ny; i++)
      m_h[i] = C0 * dt / hr[i];
  }

  template <boundary_condition bc>
  void apply_e_boundary_conditions_top ()
  {
    for (unsigned int i = 0; i < nx; i++)
    {
      const unsigned int idx = (ny - 1) * nx + i;
      const float_type next_ez = bc == boundary_condition::dirichlet ? 0.0 : ez[0 * nx + i];
      cex[idx] = (next_ez - ez[idx]) / dy;
    }
  }

  template <boundary_condition bc>
  void apply_e_boundary_conditions_right ()
  {
    for (unsigned int j = 0; j < ny; j++)
    {
      const unsigned int curr_idx = j * nx + nx - 1;
      const float_type next_ez = bc == boundary_condition::dirichlet ? 0.0 : ez[j * nx + 0];
      cey[curr_idx] = -(next_ez - ez[curr_idx]) / dx;
    }
  }

  void apply_e_boundary_conditions ()
  {
    if (top_bc == boundary_condition::dirichlet)
      apply_e_boundary_conditions_top<boundary_condition::dirichlet> ();
    else
      apply_e_boundary_conditions_top<boundary_condition::periodic> ();

    if (right_bc == boundary_condition::dirichlet)
      apply_e_boundary_conditions_right<boundary_condition::dirichlet> ();
    else
      apply_e_boundary_conditions_right<boundary_condition::periodic> ();
  }

  void update_curl_e ()
  {
    for (unsigned int j = 0; j < ny - 1; j++) /// Row index
    {
      for (unsigned int i = 0; i < nx; i++) /// Column index
      {
        const unsigned int curr_idx = (j + 0) * nx + i;
        const unsigned int next_idx = (j + 1) * nx + i;

        cex[curr_idx] = (ez[next_idx] - ez[curr_idx]) / dy;
      }
    }

    for (unsigned int j = 0; j < ny; j++)
    {
      for (unsigned int i = 0; i < nx - 1; i++)
      {
        const unsigned int curr_idx = j * nx + i;
        const unsigned int next_idx = j * nx + i + 1;
        cey[curr_idx] = -(ez[next_idx] - ez[curr_idx]) / dx;
      }
    }
  }

  void update_h ()
  {
    for (unsigned int j = 0; j < ny; j++)
    {
      for (unsigned int i = 0; i < nx; i++)
      {
        const unsigned int idx = j * nx + i;
        hx[idx] = hx[idx] - m_h[idx] * cex[idx];
      }
    }

    for (unsigned int j = 0; j < ny; j++)
    {
      for (unsigned int i = 0; i < nx; i++)
      {
        const unsigned int idx = j * nx + i;
        hy[idx] = hy[idx] - m_h[idx] * cey[idx];
      }
    }
  }

  template <boundary_condition bc>
  void apply_h_boundary_conditions_bottom ()
  {
    for (unsigned int i = 1; i < nx; i++)
    {
      const unsigned int curr_idx = 0 * nx + i;
      const unsigned int prev_idx_i = 0 * nx + i - 1;

      const float_type prev_hx_j = bc == boundary_condition::dirichlet
                                 ? 0.0
                                 : hx[(ny - 1) * nx + i];

      chz[curr_idx] = (hy[curr_idx] - hy[prev_idx_i]) / dx
                    - (hx[curr_idx] - prev_hx_j) / dy;
    }
  }

  template <boundary_condition bc>
  void apply_h_boundary_conditions_left ()
  {
    for (unsigned int j = 1; j < ny; j++)
    {
      const unsigned int curr_idx   = j * nx + 0;
      const unsigned int prev_idx_j = (j - 1) * nx + 0;

      const float_type prev_hy_i = bc == boundary_condition::dirichlet
                                 ? 0.0
                                 : hy[j * nx + nx - 1];

      chz[curr_idx] = (hy[curr_idx] - prev_hy_i) / dx
                    - (hx[curr_idx] - hx[prev_idx_j]) / dy;
    }
  }

  void apply_h_boundary_conditions ()
  {
    const float_type prev_hy_i = left_bc == boundary_condition::dirichlet ? 0.0 : hy[nx - 1];
    const float_type prev_hx_j = bottom_bc == boundary_condition::dirichlet ? 0.0 : hx[(ny - 1) * nx + 0];
    chz[0] = (hy[0] - prev_hy_i) / dx - (hx[0] - prev_hx_j) / dy;

    if (left_bc == boundary_condition::dirichlet)
      apply_h_boundary_conditions_left<boundary_condition::dirichlet> ();
    else
      apply_h_boundary_conditions_left<boundary_condition::periodic> ();

    if (bottom_bc == boundary_condition::dirichlet)
      apply_h_boundary_conditions_bottom<boundary_condition::dirichlet> ();
    else
      apply_h_boundary_conditions_bottom<boundary_condition::periodic> ();
  }

  void update_curl_h ()
  {
    for (unsigned int j = 1; j < ny; j++)
    {
      for (unsigned int i = 1; i < nx; i++)
      {
        const unsigned int curr_idx = j * nx + i;
        const unsigned int prev_idx_i = (j - 0) * nx + i - 1;
        const unsigned int prev_idx_j = (j - 1) * nx + i;

        chz[curr_idx] = (hy[curr_idx] - hy[prev_idx_i]) / dx
                      - (hx[curr_idx] - hx[prev_idx_j]) / dy;
      }
    }
  }

  void update_d ()
  {
    const float_type update_constant = C0 * dt;

    for (unsigned int j = 0; j < ny; j++)
    {
      for (unsigned int i = 0; i < nx; i++)
      {
        const unsigned int idx = j * nx + i;
        dz[idx] += update_constant * chz[idx];
      }
    }
  }

  void update_e ()
  {
    for (unsigned int j = 0; j < ny; j++)
    {
      for (unsigned int i = 0; i < nx; i++)
      {
        const unsigned int idx = j * nx + i;
        ez[idx] = dz[idx] / er[idx];
      }
    }
  }

  void calculate_cpu (unsigned int steps, const sources_holder<float_type> &s)
  {
    float_type t = 0.0;

    for (unsigned int step = 0; step < steps; t += dt, step++)
    {
      const auto begin = std::chrono::high_resolution_clock::now ();
      std::cout << "step: " << step << "; t: " << t << " ";

      update_curl_e ();
      apply_e_boundary_conditions ();
      update_h ();
      update_curl_h ();
      apply_h_boundary_conditions ();
      update_d ();

      s.update_sources (t, dz.get ());

      update_e ();

      const auto end = std::chrono::high_resolution_clock::now ();
      const std::chrono::duration<double> duration = end - begin;
      std::cout << "in " << duration.count () << "s\n";

      //if (step % 5 == 0)
      //  write_vtk ("output_" + std::to_string (step) + ".vtk", dx, dy, nx, ny, ez.get ());
    }
  }

  const float_type *get_ez () const
  {
    return ez.get ();
  }

  void calculate_gpu (unsigned int steps, const sources_holder<float_type> &s)
  {
    (void) s;

    cudaSetDevice (0);

    float_type t = 0.0;

    float_type *d_mh = nullptr;
    float_type *d_er = nullptr;
    float_type *d_ez = nullptr;
    float_type *d_dz = nullptr;
    float_type *d_hx = nullptr;
    float_type *d_hy = nullptr;

    const unsigned int sources_count = s.get_sources_count ();
    float_type *d_sources_frequencies = nullptr;
    unsigned int *d_sources_offsets = nullptr;

    cudaMalloc (&d_mh, nx * ny * sizeof (float_type));
    cudaMalloc (&d_er, nx * ny * sizeof (float_type));
    cudaMalloc (&d_ez, nx * ny * sizeof (float_type));
    cudaMalloc (&d_dz, nx * ny * sizeof (float_type));
    cudaMalloc (&d_hx, nx * ny * sizeof (float_type));
    cudaMalloc (&d_hy, nx * ny * sizeof (float_type));
    cudaMalloc (&d_sources_frequencies, sources_count * sizeof (float_type));
    cudaMalloc (&d_sources_offsets, sources_count * sizeof (unsigned int));

    cudaMemcpy (d_mh, m_h.get (), nx * ny * sizeof (float_type), cudaMemcpyHostToDevice);
    cudaMemcpy (d_er, er.get (),  nx * ny * sizeof (float_type), cudaMemcpyHostToDevice);

    cudaMemcpy (d_sources_frequencies, s.get_sources_frequencies (), sources_count * sizeof (float_type), cudaMemcpyHostToDevice);
    cudaMemcpy (d_sources_offsets, s.get_sources_offsets (), sources_count * sizeof (unsigned int), cudaMemcpyHostToDevice);

    cudaMemset (d_ez, 0, nx * ny * sizeof (float_type));
    cudaMemset (d_hx, 0, nx * ny * sizeof (float_type));
    cudaMemset (d_hy, 0, nx * ny * sizeof (float_type));

    for (unsigned int step = 0; step < steps; t += dt, step++)
    {
      const auto begin = std::chrono::high_resolution_clock::now();
      std::cout << "step: " << step << "; t: " << t << " ";

      fdtd_step (t, dt, nx, ny, dx, dy, d_mh, d_er, d_ez, d_dz, d_hx, d_hy, sources_count, d_sources_frequencies, d_sources_offsets);

      if (cudaGetLastError () != cudaSuccess)
      {
        std::cout << "Error on GPU!" << std::endl;
        return;
      }

      const auto end = std::chrono::high_resolution_clock::now ();
      const std::chrono::duration<double> duration = end - begin;
      std::cout << "in " << duration.count () << "s\n";

      if (step % 40 == 0)
      {
        cudaMemcpy (ez.get (), d_ez, nx * ny * sizeof (float_type), cudaMemcpyDeviceToHost);
        // write_vtk ("output_" + std::to_string (step) + ".vtk", dx, dy, nx, ny, ez.get ());
      }
    }

    cudaFree (d_mh);
    cudaFree (d_er);
    cudaFree (d_ez);
    cudaFree (d_dz);
    cudaFree (d_hx);
    cudaFree (d_hy);
  }

  /// Ez mode
  void calculate (unsigned int steps, const sources_holder<float_type> &s)
  {
    bool calculate_on_gpu = true;

    std::cout << "Time step: " << dt << std::endl;
    std::cout << "Nx: " << nx << "; Ny: " << ny << std::endl;

    const auto calculation_begin = std::chrono::high_resolution_clock::now ();
    if (calculate_on_gpu)
      calculate_gpu (steps, s);
    else
      calculate_cpu (steps, s);
    const auto calculation_end = std::chrono::high_resolution_clock::now ();
    const std::chrono::duration<double> duration = calculation_end - calculation_begin;
    std::cout << "Computation completed in " << duration.count () << "s\n";
  }
};

#include <QApplication>
#include "gui/main_window.h"

int main (int argc, char *argv[])
{
  const double plane_size_x = 5;

  // const double dt = 1e-22;
  const double frequency = 1e+9;
  const double lambda_min = C0 / frequency;
  const double dx = lambda_min / 20;
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

  rectangular_region_initializer top_rectangle (optimal_nx, optimal_ny, optimal_nx / 2, top_rec_bottom, slit_width, top_rec_height, er, mr);
  rectangular_region_initializer mid_rectangle (optimal_nx, optimal_ny, optimal_nx / 2, mid_rec_bottom, slit_width, distance_between_slits, er, mr);
  rectangular_region_initializer bot_rectangle (optimal_nx, optimal_ny, optimal_nx / 2, 0,              slit_width, bottom_rec_height, er, mr);

  fdtd_2d simulation (
      optimal_nx, optimal_ny,
      plane_size_x, plane_size_y,
      boundary_condition::periodic,
      boundary_condition::periodic,
      boundary_condition::periodic,
      boundary_condition::periodic,
      { &top_rectangle, &mid_rectangle, &bot_rectangle });
  // simulation.calculate (1000, soft_source);

  QApplication app (argc, argv);

  main_window window(
      optimal_nx, optimal_ny,
      static_cast<float>(plane_size_x), static_cast<float>(plane_size_y),
      [&simulation, &optimal_nx, &optimal_ny, &soft_source] (GLfloat *colors)
      {
        simulation.calculate (500, soft_source);
        auto ez = simulation.get_ez ();

        for (unsigned int j = 0; j < optimal_ny; j++)
        {
          for (unsigned int i = 0; i < optimal_nx; i++)
          {
            auto ezv = ez[j * optimal_nx + i] * 100;
            if (ezv < 0.0)
              ezv = 0.0;
            if (ezv > 1.0)
              ezv = 1.0;

            colors[3 * 4 * (j * optimal_nx + i) + 0] = ezv;
            colors[3 * 4 * (j * optimal_nx + i) + 1] = 0.0f;
            colors[3 * 4 * (j * optimal_nx + i) + 2] = 0.0f;
            colors[3 * 4 * (j * optimal_nx + i) + 3] = ezv;
            colors[3 * 4 * (j * optimal_nx + i) + 4] = 0.0f;
            colors[3 * 4 * (j * optimal_nx + i) + 5] = 0.0f;
            colors[3 * 4 * (j * optimal_nx + i) + 6] = ezv;
            colors[3 * 4 * (j * optimal_nx + i) + 7] = 0.0f;
            colors[3 * 4 * (j * optimal_nx + i) + 8] = 0.0f;
            colors[3 * 4 * (j * optimal_nx + i) + 9] = ezv;
            colors[3 * 4 * (j * optimal_nx + i) + 10] = 0.0f;
            colors[3 * 4 * (j * optimal_nx + i) + 11] = 0.0f;
          }
        }
      });
  window.resize (QSize (800, 800));
  window.show ();

  return app.exec ();
}
