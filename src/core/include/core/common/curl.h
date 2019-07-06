//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_CURL_H
#define ANYSIM_CURL_H

#include "core/common/common_defs.h"
#include "core/grid/grid.h"

/**
 * Calculate curl of Ex with periodic boundary condition
 * @param i Column index
 * @param j Row index
 */
template <typename float_type>
CPU_GPU static float_type update_curl_ex (
  unsigned int cell_id,
  const grid_topology &topology,
  const grid_geometry &geometry,
  const float_type * __restrict__ ez)
{
  const unsigned int neighbor_id = topology.get_neighbor_id (cell_id, side_to_id (side_type::top));
  return (ez[neighbor_id] - ez[cell_id]) / geometry.get_distance_between_cells_y (neighbor_id, cell_id);
}

/**
 * @param i Column index
 * @param j Row index
 */
template <typename float_type>
CPU_GPU static float_type update_curl_ey (
  unsigned int cell_id,
  const grid_topology &topology,
  const grid_geometry &geometry,
  const float_type * __restrict__ ez)
{
  const unsigned int neighbor_id = topology.get_neighbor_id (cell_id, side_to_id (side_type::right));
  return -(ez[neighbor_id] - ez[cell_id]) / geometry.get_distance_between_cells_x (neighbor_id, cell_id);
}

template <typename float_type>
CPU_GPU static float_type update_curl_h (
  unsigned int cell_id,
  const grid_topology &topology,
  const grid_geometry &geometry,
  const float_type * __restrict__ hx,
  const float_type * __restrict__ hy)
{
  // TODO For now assume that only periodic boundary conditions exist
  const unsigned int left_neighbor_id = topology.get_neighbor_id (cell_id, side_to_id (side_type::left));
  const unsigned int bottom_neighbor_id = topology.get_neighbor_id (cell_id, side_to_id (side_type::bottom));

  return (hy[cell_id] - hy[left_neighbor_id]) / geometry.get_distance_between_cells_x (cell_id, left_neighbor_id)
       - (hx[cell_id] - hx[bottom_neighbor_id]) / geometry.get_distance_between_cells_y (cell_id, bottom_neighbor_id);
}

#endif //ANYSIM_CURL_H
