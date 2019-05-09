#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
#include <cmath>

constexpr double C0 = 299792458; /// Speed of light [metres per second]

template <class float_type>
void write_vtk (
    const std::string &filename,
    const float_type dx,
    const float_type dy,
    const unsigned int nx,
    const unsigned int ny,
    const float_type *e, const float_type *h)
{
  (void)e;
  std::ofstream vtk (filename);

  vtk << "# vtk DataFile Version 3.0\n";
  vtk << "vtk output\n";
  vtk << "ASCII\n";
  vtk << "DATASET UNSTRUCTURED_GRID\n";

  vtk << "POINTS " << nx * ny * 4 << " double\n";

  for (unsigned int j = 0; j < ny; j++)
  {
    for (unsigned int i = 0; i < nx; i++)
    {
      vtk << dx * (i + 0) << " " << dy * (j + 0) << " 0.0\n";
      vtk << dx * (i + 1) << " " << dy * (j + 0) << " 0.0\n";
      vtk << dx * (i + 1) << " " << dy * (j + 1) << " 0.0\n";
      vtk << dx * (i + 0) << " " << dy * (j + 1) << " 0.0\n";
    }
  }

  vtk << "CELLS " << nx * ny << " " << nx * ny * 5 << "\n";

  for (unsigned int j = 0; j < ny; j++)
  {
    for (unsigned int i = 0; i < nx; i++)
    {
      vtk << "4 ";
      for (unsigned int k = 0; k < 4; k++)
        vtk << (j * nx + i) * 4 + k << " ";
      vtk << "\n";
    }
  }

  vtk << "CELL_TYPES " << nx * ny << "\n";
  for (unsigned int i = 0; i < nx * ny; i++)
    vtk << "9\n";

  vtk << "CELL_DATA " << nx * ny << "\n";
  vtk << "SCALARS Ey double 1\n";
  vtk << "LOOKUP_TABLE default\n";
  for (unsigned int i = 0; i < nx * ny; i++)
    vtk << e[i] << "\n";

  vtk << "SCALARS Hx double 1\n";
  vtk << "LOOKUP_TABLE default\n";
  for (unsigned int i = 0; i < nx * ny; i++)
    vtk << h[i] << "\n";
}

template <class float_type>
float_type gaussian_pulse (float_type t, float_type t_0, float_type tau)
{
  return std::exp (-(((t - t_0) / tau) * (t - t_0) / tau));
}

template <class float_type>
float_type soft_source (float_type frequency, float_type t)
{
  const float_type tau = 0.5 / frequency;
  const float_type t_0 = 6 * tau;
  return gaussian_pulse (t, t_0, tau);
}

template <class float_type>
class source
{
private:
  const float_type frequency;
  const float_type tau;
  const float_type t_0;

  const unsigned int x_position;
public:
  source () = delete;
  source (
      float_type frequency_arg, unsigned int x_position_arg)
    : frequency (frequency_arg)
    , tau (0.5 / frequency)
    , t_0 (6 * tau)
    , x_position (x_position_arg)
  { }

  unsigned int get_anchor () const
  {
    return x_position;
  }

  void update_source (float_type t, float_type *e)
  {
    const float_type source_value = gaussian_pulse (t, t_0, tau);
    e[x_position] += source_value;
  }
};

template <class float_type>
class fdtd
{
  // const float_type e0 = 8.85418781762039e-12; /// Electrical permeability in vacuum [Farad/Meter]
  // const float_type u0 = 1.25663706143592e-6;  /// Magnetic permeability in vacuum [Newton / Amper^{2}]

  const unsigned int nx, ny;
  const float_type dx, dy;
  const float_type dt;

  std::unique_ptr<float_type[]> m_ey, m_hx; /// Compute update coefficients
  std::unique_ptr<float_type[]> ey, hx;
  std::unique_ptr<float_type[]> er, hr; /// Materials properties

public:
  fdtd () = delete;
  fdtd (
      unsigned int nx_arg,
      unsigned int ny_arg,
      float_type plane_size_x,
      float_type plane_size_y)
    : nx (nx_arg)
    , ny (ny_arg)
    , dx (plane_size_x / nx)
    , dy (plane_size_y / ny)
    , dt (std::min (dx, dy) / C0 / 2.0)
    , m_ey (new float_type[nx * ny])
    , m_hx (new float_type[nx * ny])
    , ey   (new float_type[nx * ny])
    , hx   (new float_type[nx * ny])
    , er   (new float_type[nx * ny])
    , hr   (new float_type[nx * ny])
  {
    /// Assume that we are in free space
    std::fill_n (er.get (), nx * ny, 1.0);
    std::fill_n (hr.get (), nx * ny, 1.0);

    std::fill_n (hx.get (), nx * ny, 0.0);
    std::fill_n (ey.get (), nx * ny, 0.0);

    for (unsigned int i = 0; i < nx * ny; i++)
      m_ey[i] = C0 * dt / er[i];
    for (unsigned int i = 0; i < nx * ny; i++)
      m_hx[i] = C0 * dt / hr[i];
  }

  void calculate (unsigned int steps, source<float_type> &s)
  {
    std::cout << "Time step: " << dt << std::endl;
    std::cout << "NX: " << nx << std::endl;

    // float_type h_3, h_2, h_1;
    // float_type e_3, e_2, e_1;

    // h_1 = h_2 = h_3 = float_type ();
    // e_1 = e_2 = e_3 = float_type ();

    float_type t = 0.0;

    for (unsigned int step = 0; step < steps; step++)
    {
      #pragma omp parallel for
      for (unsigned int i = 0; i < nx - 1; i++)
        hx[i] += m_hx[i] * (ey[i + 1] - ey[i]) / dx;

      // PBC
      // hx[nx - 1] += m_hx[nx - 1] * (e_3 - ey[nx - 1]) / dx;
      // h_3 = h_2; h_2 = h_1; h_1 = hx[0];

      // ey[0] += m_ey[0] * (hx[0] - h_3) / dx;

      // Mirror boundary condition (ideal conductor)
      hx[nx - 1] += m_hx[nx - 1] * (0.0 - ey[nx - 1]) / dx;
      ey[0] += m_ey[0] * (hx[0]) / dx;

      // Periodic boundary condition (ideal conductor)
      // hx[nx - 1] += m_hx[nx - 1] * (ey[0] - ey[nx - 1]) / dx;
      // ey[0] += m_ey[0] * (hx[0] - hx[nx - 1]) / dx;

      #pragma omp parallel for
      for (unsigned int i = 1; i < nx; i++)
        ey[i] += m_ey[i] * (hx[i] - hx[i - 1]) / dx;

      // PBC
      // e_3 = e_2; e_2 = e_1; e_1 = ey[nx - 1];

      // Apply source
      s.update_source (t, ey.get ());

      if (step % 100 == 0)
        write_vtk ("output_" + std::to_string (step) + ".vtk", dx, dy, nx, ny, ey.get (), hx.get ());

      t += dt;
    }
  }
};

template <class float_type>
class fdtd_2d
{
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
      float_type plane_size_y)
      : nx (nx_arg)
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

    for (unsigned int i = 0; i < nx * ny; i++)
      m_e[i] = C0 * dt / er[i];
    for (unsigned int i = 0; i < nx * ny; i++)
      m_h[i] = C0 * dt / hr[i];
  }

  void update_curl_e ()
  {
    for (unsigned int j = 0; j < ny - 1; j++) /// Row index
    {
      for (unsigned int i = 0; i < nx; i++) /// Column index
      {
        const unsigned int curr_idx = (j + 0) * nx + i;
        const unsigned int next_idx = (j + 1) * nx + i;
        cex[curr_idx] = (ez[next_idx] - ez[curr_idx]) / dy; /// Cxz^{i,j}_{t} = (Ez^{i,j+1}_{t} - (Ez^{i,j}_{t})/dy
      }
    }

    /// Dirichlet boundary condition (mirror)
    for (unsigned int i = 0; i < nx; i++) /// Column index
    {
      const unsigned int idx = (ny - 1) * nx + i;
      cex[idx] = (float_type (0.0) - ez[idx]) / dy;
    }

    for (unsigned int j = 0; j < ny; j++)
    {
      unsigned int i = 0;
      for (; i < nx - 1; i++)
      {
        const unsigned int curr_idx = j * nx + i;
        const unsigned int next_idx = j * nx + i + 1;
        cey[curr_idx] = -(ez[next_idx] - ez[curr_idx]) / dx;
      }

      const unsigned int curr_idx = j * nx + i; // TODO Terrible memory access
      cey[curr_idx] = -(float_type (0.0) - ez[curr_idx]) / dx;
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
        hy[idx] = hy[idx] - m_h[idx] * cey[idx];
      }
    }
  }

  void update_curl_h ()
  {
    /// Dirichlet boundary condition (mirror)

    /// i = 1, j = 1
    chz[0] = (hy[0] - float_type (0.0)) / dx - (hx[0] - float_type (0.0)) / dy;

    /// j = 1, i > 0
    for (unsigned int i = 1; i < nx; i++)
    {
      const unsigned int curr_idx = 0 * nx + i;
      const unsigned int prev_idx_i = 0 * nx + i - 1;

      chz[curr_idx] = (hy[curr_idx] - hy[prev_idx_i]) / dx
                    - (hx[curr_idx] - float_type (0.0)) / dy;
    }

    for (unsigned int j = 1; j < ny; j++)
    {
      {
        const unsigned int curr_idx = j * nx + 0;
        const unsigned int prev_idx_j = (j - 1) * nx + 0;

        chz[curr_idx] = (hy[curr_idx] - float_type (0.0)) / dx
                      - (hx[curr_idx] - hx[prev_idx_j]) / dy;
      }

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
        float_type curl_h_z = chz[idx];
        dz[idx] += update_constant * curl_h_z;
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
        float_type d = dz[idx];
        float_type e = er[idx];
        ez[idx] = d / e;
      }
    }
  }

  void calculate (unsigned int steps, source<float_type> &s)
  {
    std::cout << "Time step: " << dt << std::endl;
    std::cout << "Nx: " << nx << "; Ny: " << ny << std::endl;

    float_type t = 0.0;

    for (unsigned int step = 0; step < steps; step++)
    {
      const auto begin = std::chrono::high_resolution_clock::now ();
      std::cout << "step: " << step << "; t: " << t << " ";

      update_curl_e ();
      update_h ();
      update_curl_h ();
      update_d ();

      s.update_source (t, dz.get ());

      update_e ();

      const auto end = std::chrono::high_resolution_clock::now ();
      const std::chrono::duration<double> duration = end - begin;
      std::cout << "in " << duration.count () << "s\n";

      if (step < 800)
      {
        if (step % 60 == 0)
          write_vtk ("output_" + std::to_string (step) + ".vtk", dx, dy, nx, ny, ez.get (), hx.get ());
      }
      else
      {
        if (step % 10 == 0)
          write_vtk ("output_" + std::to_string (step) + ".vtk", dx, dy, nx, ny, ez.get (), hx.get ());
      }

      t += dt;
    }
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

  source soft_source (frequency, (optimal_ny/2) * optimal_nx + optimal_nx / 2);

  fdtd_2d simulation (optimal_nx, optimal_ny, plane_size_x, plane_size_y);
  simulation.calculate (3000, soft_source);

  return 0;
}