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
  , buffer (new cache_padded_void_ptr[threads_count])
{
  for (unsigned int thread_id = 1; thread_id < total_threads; thread_id++)
    threads.emplace_back (&thread_pool::run_thread, this, thread_id);
}

thread_pool::~thread_pool ()
{
  finalize_pool = true;
  cv.notify_all ();

  std::for_each (threads.begin (), threads.end (), [] (std::thread &thread) { thread.join (); });
}

void thread_pool::run_thread (unsigned int thread_id)
{
  unsigned int thread_epoch = 0;

  while (!finalize_pool)
  {
    std::unique_lock guard (lock);
    cv.wait (guard, [&] {
      return epoch != thread_epoch || finalize_pool;
    });
    guard.unlock ();

    if (finalize_pool)
      return;

    thread_epoch++;
    action (thread_id, total_threads);
    barrier ();
  }
}

void thread_pool::execute (const std::function<void(unsigned int, unsigned int)> &action_arg)
{
  {
    std::lock_guard guard (lock);
    action = action_arg;
    epoch++;
  }
  cv.notify_all ();

  action (0, total_threads);
  barrier ();
}

void thread_pool::barrier ()
{
  const unsigned int thread_epoch = barrier_epoch.load ();

  if (threads_in_barrier.fetch_add (1u) == total_threads - 1)
  {
    threads_in_barrier.store (0);
    barrier_epoch.fetch_add (1u);
  }
  else
  {
    while (thread_epoch == barrier_epoch.load ())
      _mm_pause ();
  }
}