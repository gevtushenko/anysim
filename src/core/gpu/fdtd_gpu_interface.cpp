#include "core/gpu/fdtd_gpu_interface.h"
#include "core/gpu/fdtd.cuh"

template <typename float_type>
void fdtd_step(
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
    const unsigned int *sources_offsets)
{
#ifdef GPU_BUILD
  fdtd_step_gpu<float_type> (t, dt, nx, ny, dx, dy, mh, er, ez, dz, hx, hy, sources_count, sources_frequencies, sources_offsets);
#else
#endif
}

#define GEN_FDTD_INTERFACE_INSTANCE_FOR(type)                                              \
  template void fdtd_step<type>(                                                           \
      type t, type dt, unsigned int nx, unsigned int ny, type dx, type dy, const type *mh, \
      const type *er, type *ez, type *dz, type *hx, type *hy, unsigned int sources_count,  \
      const type *sources_frequencies, const unsigned int *sources_offsets);

GEN_FDTD_INTERFACE_INSTANCE_FOR (float)
GEN_FDTD_INTERFACE_INSTANCE_FOR (double)

#undef GEN_FDTD_INTERFACE_INSTANCE_FOR
