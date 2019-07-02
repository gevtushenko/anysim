#ifndef ANYSIM_REDUCE_CUH_
#define ANYSIM_REDUCE_CUH_

#include <limits>

#define FULL_WARP_MASK 0xFFFFFFFF

enum class reduce_operation
{
  sum,
  min,
  max
};

template <reduce_operation op, class out_data_type, class in_data_type>
__device__ void reduce_function (out_data_type &lhs, in_data_type rhs)
{
  switch (op)
    {
      case reduce_operation::sum: lhs = lhs + rhs; return;
      case reduce_operation::min: lhs = lhs > rhs ? rhs : lhs; return;
      case reduce_operation::max: lhs = lhs < rhs ? rhs : lhs; return;
    }
}

template <reduce_operation op, class T>
__device__ T reduce_identity_element ()
{
  switch (op)
    {
      case reduce_operation::sum: return T {};
      case reduce_operation::min: return std::numeric_limits<T>::max ();
      case reduce_operation::max: return std::numeric_limits<T>::min ();
    }

  return T {};
}

template <class T, reduce_operation op>
__device__ T warp_reduce (T val)
{
  /**
   *  For a thread at lane X in the warp, __shfl_down_sync(FULL_MASK, val, offset) gets
   *  the value of the val variable from the thread at lane X+offset of the same warp.
   *  The data exchange is performed between registers, and more efficient than going
   *  through shared memory, which requires a load, a store and an extra register to
   *  hold the address.
   */
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    reduce_function<op> (val, __shfl_down_sync (FULL_WARP_MASK, val, offset));

  return val;
}

/**
 * @brief Reduce value within block
 * @tparam T type of data to reduce
 * @tparam op reduce operation
 * @tparam warps_count blockDim.x / warpSize
 * @param val value to reduce from each fiber
 * @return reduced value on first lane of first warp
 */
template <class T, reduce_operation op, int warps_count>
__device__ T block_reduce (T val)
{
  static __shared__ T shared[warps_count]; /// Shared memory for partial results

  const int lane = threadIdx.x % warpSize;
  const int wid = threadIdx.x / warpSize;

  val = warp_reduce<T, op> (val);

  if (lane == 0) /// Main fiber stores value from it's warp
    shared[wid] = val;

  __syncthreads ();

  /// block thread id < warps count in block
  val = (threadIdx.x < blockDim.x / warpSize)
      ? shared[lane]
      : reduce_identity_element<op, T> ();

  if (wid == 0) /// Reduce within first warp
    val = warp_reduce <T, op> (val);

  return val;
}

__device__ static float atomicMax (float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    }
  while (assumed != old);
  return __int_as_float(old);
}

__device__ static float atomicMin (float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
    }
  while (assumed != old);
  return __int_as_float(old);
}

__device__ static double atomicMax (double* address, double val)
{
  unsigned long long int* address_as_i = (unsigned long long int*) address;
  unsigned long long int old = *address_as_i, assumed;
  do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed, __double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
    }
  while (assumed != old);
  return __longlong_as_double(old);
}

__device__ static double atomicMin (double* address, double val)
{
  unsigned long long int* address_as_i = (unsigned long long int*) address;
  unsigned long long int old = *address_as_i, assumed;
  do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed, __double_as_longlong(::fminf(val, __longlong_as_double(assumed))));
    }
  while (assumed != old);
  return __longlong_as_double(old);
}

template <reduce_operation op, int warps_count, class in_data_type, class out_data_type>
__device__ void block_atomic_reduce (const in_data_type *in, out_data_type* out, const unsigned int n)
{
  out_data_type result = reduce_identity_element<op, out_data_type> ();

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  while (index < n)
    {
      reduce_function<op> (result, in[index]);
      index += stride;
    }

  result = block_reduce <out_data_type, op, warps_count> (result);

  if (threadIdx.x == 0)
    {
      if (op == reduce_operation::sum)
        atomicAdd (out, result);
      else if (op == reduce_operation::min)
        atomicMin (out, result);
      else
        atomicMax (out, result);
    }
}

template <reduce_operation op, int warps_count, class in_data_type, class out_data_type>
__global__ void block_atomic_reduce_kernel (const in_data_type *in, out_data_type* out, const unsigned int n)
{
  block_atomic_reduce<op, warps_count, in_data_type, out_data_type> (in, out, n);
}

#endif
