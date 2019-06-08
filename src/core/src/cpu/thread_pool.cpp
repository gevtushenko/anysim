//
// Created by egi on 6/7/19.
//

#include "core/cpu/thread_pool.h"

#include <iostream>
#include <algorithm>

#include <xmmintrin.h> // For _mm_pause

thread_pool::thread_pool () : thread_pool (std::thread::hardware_concurrency ()) {}

thread_pool::thread_pool (unsigned int threads_count)
  : epoch (0)
  , barrier_epoch (0)
  , threads_in_barrier (0)
  , total_threads (threads_count)
{
  for (unsigned int thread_id = 1; thread_id < total_threads; thread_id++)
    threads.emplace_back (&thread_pool::run_thread, this, thread_id, total_threads);
}

thread_pool::~thread_pool ()
{
  {
    std::lock_guard guard (lock);
    finalize_pool = true;
  }

  cv.notify_all ();

  std::for_each (threads.begin (), threads.end (), [] (std::thread &thread) { thread.join (); });
}

void thread_pool::run_thread (unsigned int thread_id, unsigned int total_threads)
{
  unsigned int thread_epoch = 0;

  while (!finalize_pool)
  {
    std::unique_lock guard (lock);
    cv.wait (guard, [=] {
      return epoch.load (std::memory_order_acquire) != thread_epoch || finalize_pool;
    });
    guard.unlock ();

    if (finalize_pool)
      return;

    thread_epoch++;
    action (thread_id, total_threads);
  }
}

void thread_pool::execute (const std::function<void(unsigned int, unsigned int)> &action_arg)
{
  {
    std::lock_guard guard (lock);
    action = action_arg;
    epoch.fetch_add (1u, std::memory_order_release);
  }

  cv.notify_all ();

  action (0, total_threads);
}

void thread_pool::barrier ()
{
  const unsigned int thread_epoch = barrier_epoch.load (std::memory_order_acquire);

  const unsigned int arrived_at = threads_in_barrier.fetch_add (1u, std::memory_order_release) + 1;

  if (arrived_at != total_threads)
  {
    while (thread_epoch == barrier_epoch.load (std::memory_order_acquire))
      _mm_pause ();
  }
  else
  {
    threads_in_barrier.fetch_sub (total_threads, std::memory_order_release);
    barrier_epoch.fetch_add (1u, std::memory_order_release);
  }
}