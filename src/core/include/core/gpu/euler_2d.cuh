#ifndef EULER_2D_CUH_
#define EULER_2D_CUH_

#include <cmath>
#include <algorithm>

#include "core/common/common_defs.h"

template <class float_type>
CPU_GPU float_type speed_of_sound_in_gas (float_type gamma, float_type p, float_type rho)
{
  return std::sqrt (gamma * p / rho);
}

template <class float_type>
CPU_GPU float_type calculate_total_energy (float_type p, float_type u, float_type v, float_type rho, float_type gamma)
{
  return p / ((gamma - 1) * rho) + (u*u + v*v) / 2.0;
}

template <class float_type>
CPU_GPU void fill_state_vector (
    unsigned int i,
    float_type gamma,
    const float_type *p_rho,
    const float_type *p_u,
    const float_type *p_v,
    const float_type *p_p,
    float_type *q)
{
  const float_type rhoc = p_rho[i];
  const float_type uc   = p_u[i];
  const float_type vc   = p_v[i];
  const float_type pc   = p_p[i];

  q[0] = rhoc;
  q[1] = rhoc * uc;
  q[2] = rhoc * vc;
  q[3] = rhoc * calculate_total_energy (pc, uc, vc, rhoc, gamma);
}

template <class float_type>
CPU_GPU void fill_flux_vector (const float_type *Q_c, float_type *F_c, float_type gamma)
{
  F_c[0] = Q_c[0] * Q_c[1];
  F_c[1] = Q_c[0] * Q_c[1] * Q_c[1] + Q_c[3];
  F_c[2] = Q_c[0] * Q_c[1] * Q_c[2];
  F_c[3] = Q_c[0] * Q_c[1] * calculate_total_energy (Q_c[3], Q_c[1], Q_c[2], Q_c[0], gamma) + Q_c[1] * Q_c[3];
}

inline CPU_GPU unsigned int get_neighbor_index (
    unsigned int x, unsigned int y,
    unsigned int nx, unsigned int ny, unsigned int f)
{
  if (f == 0)
  {
    if (x == 0) // BC
      return y * nx + x;
    return y * nx + x - 1;
  }
  else if (f == 1)
  {
    if (y == 0) // BC
      return y * nx + x;
    return (y - 1) * nx + x;
  }
  else if (f == 2)
  {
    if (x == nx - 1) // BC
      return y * nx + x;
    return y * nx + x + 1;
  }
  else if (f == 3)
  {
    if (y == ny - 1) // BC
      return y * nx + x;
    return (y + 1) * nx + x;
  }

  return 0; // TODO Assert
}

template <class float_type>
CPU_GPU float_type get_normal_x (unsigned int edge)
{
  switch (edge)
  {
    case 0: return -1.0;
    case 1: return  0.0;
    case 2: return  1.0;
    case 3: return  0.0;
    default: return 0.0;
  }
}

template <class float_type>
CPU_GPU float_type get_normal_y (unsigned int edge)
{
  switch (edge)
  {
    case 0: return  0.0;
    case 1: return -1.0;
    case 2: return  0.0;
    case 3: return  1.0;
    default: return 0.0;
  }
}

template <class float_type>
CPU_GPU void rotate_vector_to_edge_coordinates (
    const unsigned int edge,
    const float_type *v,
    float_type *V)
{
  const float_type normal_x = get_normal_x<float_type> (edge);
  const float_type normal_y = get_normal_y<float_type> (edge);

  V[0] =  v[0];
  V[1] =  v[1] * normal_x + v[2] * normal_y;
  V[2] = -v[1] * normal_y + v[2] * normal_x;
  V[3] =  v[3];
}

template <class float_type>
CPU_GPU void rotate_vector_from_edge_coordinates (
    const unsigned int edge,
    const float_type *V,
    float_type *v)
{
  const float_type normal_x = get_normal_x<float_type> (edge);
  const float_type normal_y = get_normal_y<float_type> (edge);

  v[0] = V[0];
  v[1] = V[1] * normal_x - V[2] * normal_y;
  v[2] = V[1] * normal_y + V[2] * normal_x;
  v[3] = V[3];
}

template <class float_type>
CPU_GPU float_type max_speed (float_type v_c, float_type v_n, float_type u_c, float_type u_n)
{
  const float_type zero = 0.0;
  const float_type splus  = std::max(zero, std::max(u_c + v_c, u_n + v_n));
  const float_type sminus = std::min(zero, std::min(u_c - v_c, u_n - v_n));
  return std::max (splus, -sminus);
}

/**
 * Rusanov scheme calculation of numerical flux (Thierry Gallouet, Jean-Marc Herard, Nicolas Seguin,
 * Some recent Finite Volume schemes to compute Euler equations using real gas EOS, 2002):
 *
 *  \f$
 *   \phi\left(W_{L}, W_{R}\right) = \frac{F\left(W_{L}\right) + F\left(W_{R}\right)}{2} - \frac{1}{2} \lambda^{max}_{i+1/2}
 *   \lambda^{max}_{i+1/2} = max\left(\left|u_{L}\right| + c_{L}, \left|u_{R}\right| + c_{R}\right)
 *  \f$
 *
 * The idea behind this flux, insstead of approximating the exact Riemann solver, is to recall
 * that the entropy solution is the limit of viscous solution and to take a centred flux to
 * which some viscosity (with the right sight) is added (Eric Sonnendrucker,
 * Numerical methods for hyperbolic systems - lecture notes, 2013).
 *
 * @param F_sigma Rusanov flux output
 */
template <class float_type>
CPU_GPU void rusanov_scheme (
    float_type gamma,
    const float_type pc, const float_type pn,
    const float_type rhoc, const float_type rhon,
    const float_type U_c, const float_type U_n,
    const float_type *F_c, const float_type *F_n,
    const float_type *Q_n, const float_type *Q_c,
    float_type *F_sigma)
{
  for (int c = 0; c < 4; c++)
  {
    const float_type central_difference = (F_c[c] + F_n[c]) / 2;

    const float_type ss_c = speed_of_sound_in_gas (gamma, pc, rhoc);
    const float_type ss_n = speed_of_sound_in_gas (gamma, pn, rhon);

    const float_type sp = max_speed (ss_c, ss_n, U_c, U_n);
    const float_type viscosity = sp * (Q_n[c] - Q_c[c]) / 2;

    F_sigma[c] = central_difference - viscosity;
  }
}

/**
 * Calculate pressure for ideal gas (Majid Ahmadi, Wahid S. Ghaly, A Finite Volume for the
 * two-dimensional euler equations with solution adaptation on unstructured meshes)
 */
template <class float_type>
CPU_GPU float_type calculate_p (float_type gamma, float_type E, float_type u, float_type v, float_type rho)
{
  return (E - (u * u + v * v) / 2) * (gamma - 1) * rho;
}

template <class float_type>
float_type euler_2d_calculate_dt_gpu (
    unsigned int n_cells,
    float_type gamma,
    float_type cfl,
    float_type min_len,
    float_type *workspace,
    const float_type *p_rho,
    const float_type *p_u,
    const float_type *p_v,
    const float_type *p_p);

template <class float_type>
void euler_2d_calculate_next_time_step_gpu (
    unsigned int nx,
    unsigned int ny,
    float_type dt,
    float_type gamma,
    float_type cell_area,
    const float_type *edge_length,
    const float_type *p_rho,
    float_type *p_rho_next,
    const float_type *p_u,
    float_type *p_u_next,
    const float_type *p_v,
    float_type *p_v_next,
    const float_type *p_p,
    float_type *p_p_next);

#endif
