#include "core/gpu/fdtd_gpu_interface.h"
#include "core/gpu/fdtd.cuh"
#include "cpp/common_funcs.h"

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
    const unsigned int *sources_offsets)
{
#ifdef GPU_BUILD
  fdtd_step_gpu<float_type> (t, C0_p_dt, topology, geometry, mh, er, ez, dz, hx, hy, sources_count, sources_frequencies, sources_offsets);
#else
  cpp_unreferenced (t, C0_p_dt, topology, geometry, mh, er, ez, dz, hx, hy, sources_count, sources_frequencies, sources_offsets);
#endif
}

#define GEN_FDTD_INTERFACE_INSTANCE_FOR(type)                                             \
  template void fdtd_step<type>(                                                          \
      type, type, const grid_topology &, const grid_geometry &, const type *mh,           \
      const type *er, type *ez, type *dz, type *hx, type *hy, unsigned int sources_count, \
      const type *sources_frequencies, const unsigned int *sources_offsets);

GEN_FDTD_INTERFACE_INSTANCE_FOR (float)
GEN_FDTD_INTERFACE_INSTANCE_FOR (double)

#undef GEN_FDTD_INTERFACE_INSTANCE_FOR
