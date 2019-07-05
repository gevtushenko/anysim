//
// Created by egi on 6/15/19.
//

#ifndef ANYSIM_SOLVER_H
#define ANYSIM_SOLVER_H

class grid;
class workspace;
class configuration;

class solver
{
public:
  solver () = delete;
  explicit solver (
      thread_pool &threads_arg,
      workspace &solver_workspace_arg)
    : threads (threads_arg)
    , solver_workspace (solver_workspace_arg)
  { }

  virtual ~solver () = default;

  /// Computes one step
  virtual double solve (unsigned int step, unsigned int thread_id, unsigned int total_threads) = 0;
  virtual void apply_configuration (const configuration &config, std::size_t config_id, grid *solver_grid, int gpu_num) = 0;
  virtual void handle_grid_change () = 0;
  virtual void fill_configuration_scheme (configuration &config, std::size_t config_id) = 0;
  virtual bool is_gpu_supported () const = 0;

protected:
  thread_pool &threads;
  workspace &solver_workspace;
};

#endif //ANYSIM_SOLVER_H
