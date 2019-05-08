#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>

template <class float_type>
void write_vtk (const std::string &filename, const float_type dx, const unsigned int nx, const float_type *e, const float_type *h)
{
  (void)e;
  std::ofstream vtk (filename);

  vtk << "# vtk DataFile Version 3.0\n";
  vtk << "vtk output\n";
  vtk << "ASCII\n";
  vtk << "DATASET UNSTRUCTURED_GRID\n";

  vtk << "POINTS " << nx * 4 << " double\n";

  const float_type dy = nx * dx / 2;
  for (unsigned int i = 0; i < nx; i++)
  {
    vtk << dx * (i + 0) << " " << 0  << " 0.0\n";
    vtk << dx * (i + 1) << " " << 0  << " 0.0\n";
    vtk << dx * (i + 1) << " " << dy << " 0.0\n";
    vtk << dx * (i + 0) << " " << dy << " 0.0\n";
  }

  vtk << "CELLS " << nx << " " << nx * 5 << "\n";

  for (unsigned int i = 0; i < nx; i++)
  {
    vtk << "4 ";
    for (unsigned int j = 0; j < 4; j++)
      vtk << i * 4 + j << " ";
    vtk << "\n";
  }

  vtk << "CELL_TYPES " << nx << "\n";
  for (unsigned int i = 0; i < nx; i++)
    vtk << "9\n";

  vtk << "CELL_DATA " << nx << "\n";
  vtk << "SCALARS Ey double 1\n";
  vtk << "LOOKUP_TABLE default\n";
  for (unsigned int i = 0; i < nx; i++)
    vtk << e[i] << "\n";

  vtk << "SCALARS Hx double 1\n";
  vtk << "LOOKUP_TABLE default\n";
  for (unsigned int i = 0; i < nx; i++)
    vtk << h[i] << "\n";
}

template <class float_type>
class fdtd
{
  const float_type e0 = 8.85418781762039e-12; /// Electrical permeability in vacuum [Farad/Meter]
  const float_type u0 = 1.25663706143592e-6;  /// Magnetic permeability in vacuum [Newton / Amper^{2}]
  const float_type c0;          /// Speed of light [Meter/Second]

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
    : c0 (std::sqrt (1.0 / (e0 * u0)))
    , nx (nx_arg)
    , ny (ny_arg)
    , dx (plane_size_x / nx)
    , dy (plane_size_y / ny)
    , dt (std::min (dx, dy) / c0 / 2.0)
    , m_ey (new float_type[nx * ny])
    , m_hx (new float_type[nx * ny])
    , ey   (new float_type[nx * ny])
    , hx   (new float_type[nx * ny])
    , er   (new float_type[nx * ny])
    , hr   (new float_type[nx * ny])
  {
    // TODO For 2D
    // dt = 1.0 / (c0 * std::sqrt (1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz)))

    /// Assume that we are in free space
    std::fill_n (er.get (), nx * ny, 1.0);
    std::fill_n (hr.get (), nx * ny, 1.0);

    std::fill_n (hx.get (), nx * ny, 0.0);
    std::fill_n (ey.get (), nx * ny, 0.0);

    for (unsigned int i = 0; i < nx * ny; i++)
      m_ey[i] = c0 * dt / er[i];
    for (unsigned int i = 0; i < nx * ny; i++)
      m_hx[i] = c0 * dt / hr[i];
  }

  void calculate (unsigned int steps)
  {
    std::cout << "Time step: " << dt << std::endl;

    float_type h_3, h_2, h_1;
    float_type e_3, e_2, e_1;

    h_1 = h_2 = h_3 = float_type ();
    e_1 = e_2 = e_3 = float_type ();

    for (unsigned int step = 0; step < steps; step++)
    {
      if (step == 10)
        ey[nx / 2] += 1.0;

      for (unsigned int i = 0; i < nx - 1; i++)
        hx[i] += m_hx[i] * (ey[i + 1] - ey[i]) / dx;

      // PBC
      hx[nx - 1] += m_hx[nx - 1] * (e_3 - ey[nx - 1]) / dx;
      h_3 = h_2; h_2 = h_1; h_1 = hx[0];

      ey[0] += m_ey[0] * (hx[0] - h_3) / dx;

      // Mirror boundary condition (ideal conductor)
      // hx[nx - 1] += m_hx[nx - 1] * (0.0 - ey[nx - 1]) / dx;
      // ey[0] += m_ey[0] * (hx[0]) / dx;

      // Periodic boundary condition (ideal conductor)
      // hx[nx - 1] += m_hx[nx - 1] * (ey[0] - ey[nx - 1]) / dx;
      // ey[0] += m_ey[0] * (hx[0] - hx[nx - 1]) / dx;

      // Perfect boundary condition


      for (unsigned int i = 1; i < nx; i++)
        ey[i] += m_ey[i] * (hx[i] - hx[i - 1]) / dx;

      // PBC
      e_3 = e_2; e_2 = e_1; e_1 = ey[nx - 1];

      write_vtk ("output_" + std::to_string (step) + ".vtk", dx, nx, ey.get (), hx.get ());
    }
  }
};

int main()
{
  // const double dt = 1e-22;
  const double plane_size_x = 1e-10;
  const double plane_size_y = 1e-10;

  fdtd simulation (300, 1, plane_size_x, plane_size_y);
  simulation.calculate (1000);

  return 0;
}