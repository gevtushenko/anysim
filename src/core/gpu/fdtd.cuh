//
// Created by egi on 5/10/19.
//

#ifndef FDTD_FDTD_GPU_H
#define FDTD_FDTD_GPU_H

template <typename float_type>
void fdtd_step_gpu (
    float_type t,
    float_type dt,
    unsigned int nx, unsigned int ny,
    float_type dx, float_type dy,
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
