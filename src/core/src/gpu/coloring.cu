#include "core/gpu/coloring.cuh"
#include "core/gpu/reduce.cuh"

#include <cuda_runtime.h>
#include <algorithm>

template <typename float_type>
__global__ void fill_colors_kernel (unsigned int cells_number, const float_type *ez, float *colors, const float *min_max)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  const float min_value = min_max[0];
  const float max_value = min_max[1];

  if (idx < cells_number)
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
void fill_colors (unsigned int cells_number, const float_type *ez, float *colors, const float *min_max)
{
  if (!colors)
    return;

  dim3 block_size = dim3 (512);
  dim3 grid_size;

  grid_size.x = (cells_number + block_size.x - 1) / block_size.x;

  fill_colors_kernel<<<grid_size, block_size>>> (cells_number, ez, colors, min_max);
  cudaDeviceSynchronize ();
}

template <typename float_type>
void find_min_max (unsigned int cells_number, const float_type *data, float *min_max)
{
  constexpr int warps_per_block = 32;
  constexpr int warp_size = 32;
  constexpr int threads_per_block = warps_per_block * warp_size;

  cudaMemset (min_max, 0, 2 * sizeof (float));

  int blocks = std::min ((cells_number + threads_per_block - 1) / threads_per_block, 1024u);
  block_atomic_reduce_kernel<reduce_operation::min, warps_per_block> <<<blocks, threads_per_block>>> (data, min_max + 0, cells_number);
  block_atomic_reduce_kernel<reduce_operation::max, warps_per_block> <<<blocks, threads_per_block>>> (data, min_max + 1, cells_number);
}

template void fill_colors<float>  (unsigned int cells_number, const float *ez, float *colors, const float *min_max);
template void fill_colors<double> (unsigned int cells_number, const double *ez, float *colors, const float *min_max);

template void find_min_max<float>  (unsigned int cells_number, const float *ez, float *min_max);
template void find_min_max<double> (unsigned int cells_number, const double *ez, float *min_max);
