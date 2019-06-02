#include "core/gpu/coloring.cuh"

#include <cuda_runtime.h>

template <typename float_type>
__global__ void fill_colors_kernel (unsigned int nx, unsigned int ny, const float_type *ez, float *colors)
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

template <typename float_type>
void fill_colors (unsigned int nx, unsigned int ny, const float_type *ez, float *colors)
{
  if (!colors)
    return;

  dim3 block_size = dim3 (32, 32);
  dim3 grid_size;

  grid_size.x = (nx + block_size.x - 1) / block_size.x;
  grid_size.y = (ny + block_size.y - 1) / block_size.y;

  // TODO Calculate block sizes
  fill_colors_kernel<<<grid_size, block_size>>> (nx, ny, ez, colors);
  cudaDeviceSynchronize ();
}

template void fill_colors<float>  (unsigned int nx, unsigned int ny, const float *ez, float *colors);
template void fill_colors<double> (unsigned int nx, unsigned int ny, const double *ez, float *colors);
