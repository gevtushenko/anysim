//
// Created by egi on 6/7/19.
//

#ifndef ANYSIM_THREAD_POOL_H
#define ANYSIM_THREAD_POOL_H

#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <cstddef>
#include <functional>
#include <condition_variable>

constexpr inline unsigned int get_level1_dcache_linesize ()
{
  return 64;
}

struct cache_padded_void_ptr
{
  void *ptr;
  std::byte padding[get_level1_dcache_linesize () - sizeof (void *)];
};

class work_range
{
public:
  work_range (unsigned int chunk_begin_arg, unsigned int chunk_end_arg)
  : chunk_begin (chunk_begin_arg)
  , chunk_end (chunk_end_arg) { }

  work_range (const work_range &range) = default;

  static work_range split (unsigned int work_range, unsigned int thread_id, unsigned int total_threads)
  {
    const unsigned int avg_work_per_thread = work_range / total_threads;
    const unsigned int thread_beg = avg_work_per_thread * thread_id;
    const unsigned int thread_end = thread_id == total_threads - 1
                                  ? work_range
                                  : avg_work_per_thread * (thread_id + 1);
    return { thread_beg, thread_end };
  }

  const unsigned int chunk_begin;
  const unsigned int chunk_end;
};

constexpr inline unsigned int get_main_thread ()
{
  return 0;
}

inline bool is_main_thread (unsigned int thread_id)
{
  return thread_id == get_main_thread ();
}

class thread_pool
{
public:
  thread_pool (const thread_pool &) = delete;

  thread_pool ();
  explicit thread_pool (unsigned int thread_count);
  ~thread_pool ();

  void execute (const std::function<void(unsigned int, unsigned int)> &action_arg);

  void barrier ();

  template <class data_type>
  void reduce_min (unsigned int thread_id, data_type &value)
  {
    static_assert (std::is_copy_assignable<data_type>::value, "Error! Data type in reduce function has to be copy assignable");

    buffer[thread_id].ptr = &value;
    barrier ();

    if (is_main_thread (thread_id))
      for (unsigned int thread = get_main_thread () + 1; thread < total_threads; thread++)
        if (*reinterpret_cast<data_type*> (buffer[thread].ptr) < *reinterpret_cast<data_type*> (buffer[thread_id].ptr))
          buffer[get_main_thread ()].ptr = buffer[thread].ptr;

    barrier ();
    value = *reinterpret_cast<data_type*> (buffer[get_main_thread ()].ptr);
    barrier ();
  }

  template <class data_type>
  void reduce_max (unsigned int thread_id, data_type &value)
  {
    static_assert (std::is_copy_assignable<data_type>::value, "Error! Data type in reduce function has to be copy assignable");

    buffer[thread_id].ptr = &value;
    barrier ();

    if (is_main_thread (thread_id))
      for (unsigned int thread = get_main_thread () + 1; thread < total_threads; thread++)
        if (*reinterpret_cast<data_type*> (buffer[thread].ptr) > *reinterpret_cast<data_type*> (buffer[thread_id].ptr))
          buffer[get_main_thread ()].ptr = buffer[thread].ptr;

    barrier ();
    value = *reinterpret_cast<data_type*> (buffer[get_main_thread ()].ptr);
    barrier ();
  }

private:
  void run_thread (unsigned int thread_id);

private:
  std::mutex lock;
  unsigned int epoch;
  std::condition_variable cv;
  std::vector<std::thread> threads;

  std::atomic<unsigned int> barrier_epoch;
  std::atomic<unsigned int> threads_in_barrier;

  bool finalize_pool = false;
  const unsigned int total_threads;
  std::function<void(unsigned int, unsigned int)> action;
  std::unique_ptr<cache_padded_void_ptr[]> buffer;
};

#endif //ANYSIM_THREAD_POOL_H
