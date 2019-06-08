//
// Created by egi on 6/7/19.
//

#ifndef ANYSIM_THREAD_POOL_H
#define ANYSIM_THREAD_POOL_H

#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>

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

inline bool is_main_thread (unsigned int thread_id)
{
  return thread_id == 0;
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
};

#endif //ANYSIM_THREAD_POOL_H
