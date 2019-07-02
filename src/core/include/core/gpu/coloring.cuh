#ifndef FDTD_COLORING_GPU_CUH
#define FDTD_COLORING_GPU_CUH

#include "core/common/common_defs.h"

#include <cmath>

/*
 * H(Hue): 0 - 360 degree (integer)
 * S(Saturation): 0 - 1.00 (double)
 * V(Value): 0 - 1.00 (double)
 */
inline CPU_GPU float map (float x, float in_min, float in_max, float out_min, float out_max)
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

inline CPU_GPU void hsv_to_rgb (int h, double s, double v, float output[3])
{
  float c = s * v;
  float x = c * (1 - std::abs (fmodf (h / 60.0, 2) - 1));
  float m = v - c;
  float Rs, Gs, Bs;

  if (h >= 0 && h < 60)
  {
    Rs = c;
    Gs = x;
    Bs = 0;
  }
  else if (h >= 60 && h < 120)
  {
    Rs = x;
    Gs = c;
    Bs = 0;
  }
  else if (h >= 120 && h < 180)
  {
    Rs = 0;
    Gs = c;
    Bs = x;
  }
  else if (h >= 180 && h < 240)
  {
    Rs = 0;
    Gs = x;
    Bs = c;
  }
  else if (h >= 240 && h < 300)
  {
    Rs = x;
    Gs = 0;
    Bs = c;
  }
  else
  {
    Rs = c;
    Gs = 0;
    Bs = x;
  }

  output[0] = Rs + m;
  output[1] = Gs + m;
  output[2] = Bs + m;
}

inline void fill_vertex_color (float ez, float *colors, float min_field_value, float max_field_value)
{
  int hue = map (ez, min_field_value, max_field_value, 0.0, 360.0);
  hsv_to_rgb (hue, 0.6, 1.0, colors);
}

template <typename float_type>
void fill_colors (unsigned int cells_number, const float_type *ez, float *colors, const float *min_max);

template <typename float_type>
void find_min_max (unsigned int cells_number, const float_type *ez, float *min_max);

#endif // FDTD_COLORING_GPU_H
