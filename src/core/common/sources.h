//
// Created by egi on 5/10/19.
//

#ifndef FDTD_SOURCES_H
#define FDTD_SOURCES_H

#include "core/common/common_defs.h"

template <class float_type>
CPU_GPU float_type gaussian_pulse (float_type t, float_type t_0, float_type tau)
{
  return std::exp (-(((t - t_0) / tau) * (t - t_0) / tau));
}

template <class float_type>
CPU_GPU float_type calculate_source (float_type t, float_type frequency)
{
  const float_type tau = 0.5 / frequency;
  const float_type t_0 = 6 * tau;
  return gaussian_pulse (t, t_0, tau);
}

#endif //FDTD_SOURCES_H
