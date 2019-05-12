#include "core/gpu/coloring.cuh"

#include <cuda_runtime.h>

/*
 * H(Hue): 0 - 360 degree (integer)
 * S(Saturation): 0 - 1.00 (double)
 * V(Value): 0 - 1.00 (double)
 */
__device__ float map (float x, float in_min, float in_max, float out_min, float out_max)
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

__device__ void hsv_to_rgb (int h, double s, double v, float output[3])
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

__device__ void fill_vertex_color (float ez, float *colors)
{
  int hue = map (ez, -0.01, 0.01, 180.0, 360.0);
  hsv_to_rgb (hue, 0.6, 1.0, colors);
}

__global__ void fill_colors_kernel (unsigned int nx, unsigned int ny, const double *ez, float *colors)
{
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int idx = j * nx + i;

  if (idx < nx * ny)
  {
    double ezv = ez[idx];
    float *color = colors + 4 * 3 * idx;

    int hue = map (ezv, -0.01, 0.01, 180.0, 360.0);

    hsv_to_rgb (hue, 0.6, 1.0, color + 0);
    hsv_to_rgb (hue, 0.6, 1.0, color + 3);
    hsv_to_rgb (hue, 0.6, 1.0, color + 6);
    hsv_to_rgb (hue, 0.6, 1.0, color + 9);
  }
}

void fill_colors (unsigned int nx, unsigned int ny, const double *ez, float *colors)
{
  if (!colors)
    return;

  dim3 block_size = dim3 (32, 32);
  dim3 grid_size;

  grid_size.x = (nx + block_size.x - 1) / block_size.x;
  grid_size.y = (ny + block_size.y - 1) / block_size.y;

  // TODO Calculate block sizes
  fill_colors_kernel<<<grid_size, block_size>>> (nx, ny, ez, colors);
}
