//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_CURL_H
#define ANYSIM_CURL_H

#include "core/common/common_defs.h"

/**
 * Calculate curl of Ex with periodic boundary condition
 * @param i Column index
 * @param j Row index
 */
template <typename float_type>
CPU_GPU static float_type update_curl_ex (
  unsigned int i, unsigned int j,
  const unsigned int nx,
  const unsigned int ny,
  const float_type dy,
  const float_type * __restrict__ ez)
{
  // TODO For now assume that only periodic boundary conditions exist
  const unsigned int curr_idx   = (j + 0) * nx + i;
  const unsigned int next_idx_j = j < ny - 1 ? (j + 1) * nx + i : 0 * nx + i;
  return (ez[next_idx_j] - ez[curr_idx]) / dy;
}

/**
 * @param i Column index
 * @param j Row index
 */
template <typename float_type>
CPU_GPU static float_type update_curl_ey (
  unsigned int i, unsigned int j,
  const unsigned int nx,
  const float_type dx,
  const float_type * __restrict__ ez)
{
  // TODO For now assume that only periodic boundary conditions exist
  const unsigned int curr_idx   = (j + 0) * nx + i;
  const unsigned int next_idx_i = i < nx - 1 ? j * nx + i + 1 : j * nx + 0;
  return -(ez[next_idx_i] - ez[curr_idx]) / dx;
}

template <typename float_type>
CPU_GPU static float_type update_curl_h (
  unsigned int i,
  unsigned int j,
  const unsigned int nx,
  const unsigned int ny,
  const float_type dx,
  const float_type dy,
  const float_type * __restrict__ hx,
  const float_type * __restrict__ hy)
{
  // TODO For now assume that only periodic boundary conditions exist
  const unsigned int curr_idx   = (j + 0) * nx + i;
  const unsigned int prev_idx_i = i > 0 ? (j + 0) * nx + i - 1 : j * nx + nx - 1;
  const unsigned int prev_idx_j = j > 0 ? (j - 1) * nx + i     : (ny - 1) * nx + i;

  return (hy[curr_idx] - hy[prev_idx_i]) / dx
       - (hx[curr_idx] - hx[prev_idx_j]) / dy;
}

#endif //ANYSIM_CURL_H
