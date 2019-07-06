//
// Created by egi on 5/10/19.
//

#ifndef FDTD_FDTD_GPU_H
#define FDTD_FDTD_GPU_H

#include "core/common/common_defs.h"
#include "core/common/curl.h"

template <class float_type>
CPU_GPU void fdtd_2d_update_h (
    unsigned int cell_id,
    const grid_topology &topology,
    const grid_geometry &geometry,

    const float_type * __restrict__ ez,
    const float_type * __restrict__ mh,
    float_type * __restrict__ hx,
    float_type * __restrict__ hy)
{
  const float_type cex = update_curl_ex (cell_id, topology, geometry, ez);
  const float_type cey = update_curl_ey (cell_id, topology, geometry, ez);

  // update_h
  hx[cell_id] -= mh[cell_id] * cex;
  hy[cell_id] -= mh[cell_id] * cey;
}

template <class float_type>
CPU_GPU void fdtd_2d_update_e (
    unsigned int cell_id,
    const float_type t,
    const float_type C0_p_dt,
    const grid_topology &topology,
    const grid_geometry &geometry,

    const float_type * __restrict__ er,
    const float_type * __restrict__ hx,
    const float_type * __restrict__ hy,

    float_type * __restrict__ dz,
    float_type * __restrict__ ez,

    unsigned int sources_count,
    const float_type * __restrict__ sources_frequencies,
    const unsigned int * __restrict__ sources_offsets)
{
  const float_type chz = update_curl_h (cell_id, topology, geometry, hx, hy);

  dz[cell_id] += C0_p_dt * chz; // update d = C0 * dt * curl Hz

  // TODO Extract into separate kernel
  for (unsigned int source_id = 0; source_id < sources_count; source_id++)
    if (sources_offsets[source_id] == cell_id)
      dz[cell_id] += calculate_source (t, sources_frequencies[source_id]);

  ez[cell_id] = dz[cell_id] / er[cell_id]; // update e
}

template <typename float_type>
void fdtd_step_gpu (
    float_type t,
    float_type C0_p_dt,
    const grid_topology &topology,
    const grid_geometry &geometry,
    const float_type *mh,
    const float_type *er,
    float_type *ez,
    float_type *dz,
    float_type *hx,
    float_type *hy,
    unsigned int sources_count,
    const float_type *sources_frequencies,
    const unsigned int *sources_offsets);

#endif //FDTD_FDTD_GPU_H
