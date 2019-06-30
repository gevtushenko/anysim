#include "core/gpu/coloring.cuh"
#include "core/gpu/reduce.cuh"

#include <cuda_runtime.h>
#include <algorithm>

template <typename float_type>
__global__ void fill_colors_kernel (unsigned int nx, unsigned int ny, const float_type *ez, float *colors, const float *min_max)
{
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int idx = j * nx + i;

  const float min_value = min_max[0];
  const float max_value = min_max[1];

  if (idx < nx * ny)
  {
    double ezv = ez[idx];
    float *color = colors + 4 * 3 * idx;

    int hue = map (ezv, min_value, max_value, 0.0, 360.0);

    hsv_to_rgb (hue, 0.6, 1.0, color + 0);
    hsv_to_rgb (hue, 0.6, 1.0, color + 3);
    hsv_to_rgb (hue, 0.6, 1.0, color + 6);
    hsv_to_rgb (hue, 0.6, 1.0, color + 9);
  }
}

template <typename float_type>
void fill_colors (unsigned int nx, unsigned int ny, const float_type *ez, float *colors, const float *min_max)
{
  if (!colors)
    return;

  dim3 block_size = dim3 (32, 32);
  dim3 grid_size;

  grid_size.x = (nx + block_size.x - 1) / block_size.x;
  grid_size.y = (ny + block_size.y - 1) / block_size.y;

  fill_colors_kernel<<<grid_size, block_size>>> (nx, ny, ez, colors, min_max);
  cudaDeviceSynchronize ();
}

template <typename float_type>
void find_min_max (unsigned int nx, unsigned int ny, const float_type *data, float *min_max)
{
  constexpr int warps_per_block = 32;
  constexpr int warp_size = 32;
  constexpr int threads_per_block = warps_per_block * warp_size;

  cudaMemset (min_max, 0, 2 * sizeof (float));

  int blocks = std::min ((nx * ny + threads_per_block - 1) / threads_per_block, 1024u);
  block_atomic_reduce<reduce_operation::min, warps_per_block> <<<blocks, threads_per_block>>> (data, min_max + 0, nx * ny);
  block_atomic_reduce<reduce_operation::max, warps_per_block> <<<blocks, threads_per_block>>> (data, min_max + 1, nx * ny);
}

template void fill_colors<float>  (unsigned int nx, unsigned int ny, const float *ez, float *colors, const float *min_max);
template void fill_colors<double> (unsigned int nx, unsigned int ny, const double *ez, float *colors, const float *min_max);

template void find_min_max<float>  (unsigned int nx, unsigned int ny, const float *ez, float *min_max);
template void find_min_max<double> (unsigned int nx, unsigned int ny, const double *ez, float *min_max);
