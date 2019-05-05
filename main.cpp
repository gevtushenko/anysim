#include <iostream>
#include <memory>
#include <cmath>

template <class float_type>
class fdtd
{
  const float_type e0 = 8.85418781762039e-12; /// Electrical permeability in vacuum [Farad/Meter]
  const float_type u0 = 1.25663706143592e-6;  /// Magnetic permeability in vacuum [Newton / Amper^{2}]
  const float_type c0;          /// Speed of light [Meter/Second]
  const float_type dt;

  const unsigned int nx, ny;
  const float_type dx, dy;

  std::unique_ptr<float_type[]> m_ey, m_hx; /// Compute update coefficients
  std::unique_ptr<float_type[]> ey, hx;
  std::unique_ptr<float_type[]> er, hr; /// Materials properties

public:
  fdtd () = delete;
  fdtd (
      unsigned int nx_arg,
      unsigned int ny_arg,
      float_type dt_arg,
      float_type plane_size_x,
      float_type plane_size_y)
    : c0 (std::sqrt (1.0 / (e0 * u0)))
    , dt (dt_arg)
    , nx (nx_arg)
    , ny (ny_arg)
    , dx (plane_size_x / nx)
    , dy (plane_size_y / ny)
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
      m_ey[i] = c0 * dt / er[i];
    for (unsigned int i = 0; i < nx * ny; i++)
      m_hx[i] = c0 * dt / hr[i];

    // tst
    ey[5] = 1.0;
  }

  void calculate (unsigned int steps)
  {
    for (unsigned int step = 0; step < steps; step++)
    {
      for (unsigned int i = 0; i < nx - 1; i++)
        hx[i] += m_hx[i] * (ey[i + 1] - ey[i]) / dx;
      hx[nx - 1] += m_hx[nx - 1] * (0.0 - ey[nx - 1]) / dx;

      ey[0] += m_ey[0] * (hx[0]) / dx;
      for (unsigned int i = 1; i < nx; i++)
        ey[i] += m_ey[i] * (hx[i] - hx[i - 1]) / dx;

      std::cout << "E" << step << ": ";
      for (unsigned int i = 0; i < nx; i++)
        std::cout << ey[i] << " ";
      std::cout << "\n";
    }
  }
};

int main()
{
  const double dt = 1e-30;
  const double plane_size_x = 1e-10;
  const double plane_size_y = 1e-10;

  fdtd simulation (10, 1, dt, plane_size_x, plane_size_y);
  simulation.calculate (10);

  return 0;
}