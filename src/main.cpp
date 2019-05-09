#include <iostream>
#include <chrono>
#include <memory>
#include <cmath>

#include "core/sources.h"
#include "io/vtk.h"

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

  virtual void fill_region (float_type *er, float_type *hr) = 0;

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
      float_type er_value_arg,
      float_type hr_value_arg)
    : region_initializer<float_type> (nx, ny)
    , region_i_start (region_i_start_arg)
    , region_j_start (region_j_start_arg)
    , width (width_arg)
    , er_value (er_value_arg)
    , hr_value (hr_value_arg)
  { }

  void fill_region (float_type *er, float_type *hr) final
  {
    using ri = region_initializer<float_type>;

    for (unsigned int j = region_j_start; j < region_j_start + width; j++)
    {
      for (unsigned int i = region_i_start; i < region_i_start + width; i++)
      {
        er[j * ri::nx + i] = er_value;
        hr[j * ri::nx + i] = hr_value;
      }
    }
  }

protected:
  const unsigned int region_i_start, region_j_start, width;
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
      region_initializer<float_type> &initializer)
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

    initializer.fill_region (er.get (), hr.get ());

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

  /// Ez mode
  void calculate (unsigned int steps, sources_holder<float_type> &s)
  {
    std::cout << "Time step: " << dt << std::endl;
    std::cout << "Nx: " << nx << "; Ny: " << ny << std::endl;

    float_type t = 0.0;

    const auto calculation_begin = std::chrono::high_resolution_clock::now ();
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

      if (step % 5 == 0)
        write_vtk ("output_" + std::to_string (step) + ".vtk", dx, dy, nx, ny, ez.get ());
    }

    const auto calculation_end = std::chrono::high_resolution_clock::now ();
    const std::chrono::duration<double> duration = calculation_end - calculation_begin;
    std::cout << "Computation completed in " << duration.count () << "s\n";
  }
};

int main()
{
  const double plane_size_x = 5;

  // const double dt = 1e-22;
  const double frequency = 2e+9;
  const double lambda_min = C0 / frequency;
  const double dx = lambda_min / 15;
  const auto optimal_nx = static_cast<unsigned int> (std::ceil (plane_size_x / dx));
  const auto optimal_ny = optimal_nx;
  const double plane_size_y = dx * optimal_ny;

  sources_holder<double> soft_source;

  for (unsigned int j = 0; j < 3; j++)
    for (unsigned int i = 0; i < 3; i++)
      soft_source.append_source (frequency, ((j + 1) * optimal_ny/4) * optimal_nx + (i + 1) * optimal_nx / 4);

  rectangular_region_initializer rectangle (optimal_nx, optimal_ny, optimal_nx / 2, optimal_ny / 4, optimal_nx / 2, 2.0, 2.0);

  fdtd_2d simulation (
      optimal_nx, optimal_ny,
      plane_size_x, plane_size_y,
      boundary_condition::dirichlet,
      boundary_condition::dirichlet,
      boundary_condition::dirichlet,
      boundary_condition::dirichlet,
      rectangle);
  simulation.calculate (1000, soft_source);

  return 0;
}