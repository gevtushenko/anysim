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
  void run_thread (unsigned int thread_id, unsigned int total_threads);

private:
  std::mutex lock;
  std::condition_variable cv;
  std::atomic<unsigned int> epoch;
  std::vector<std::thread> threads;

  std::atomic<unsigned int> barrier_epoch;
  std::atomic<unsigned int> threads_in_barrier;

  bool finalize_pool = false;
  const unsigned int total_threads;
  std::function<void(unsigned int, unsigned int)> action;
};

#endif //ANYSIM_THREAD_POOL_H
