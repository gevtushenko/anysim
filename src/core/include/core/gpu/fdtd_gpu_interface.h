//
// Created by egi on 5/10/19.
//

#ifndef FDTD_FDTD_GPU_INTERFACE_H
#define FDTD_FDTD_GPU_INTERFACE_H

class grid_topology;
class grid_geometry;

template <typename float_type>
void fdtd_step(
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

#endif //FDTD_FDTD_GPU_INTERFACE_H
