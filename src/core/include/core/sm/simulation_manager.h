//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_SIMULATION_MANAGER_H
#define ANYSIM_SIMULATION_MANAGER_H

#include <string>
#include <memory>

#include "core/cpu/thread_pool.h"

class grid;
class solver;
class workspace;
class configuration;
class result_extractor;

class simulation_manager
{
public:
  simulation_manager () = delete;
  simulation_manager (
      const std::string &solver_arg,
      double max_simulation_time_arg,
      bool use_double_precision_arg,
      workspace &workspace_arg);
  void fill_configuration_scheme (configuration &scheme, std::size_t scheme_id);
  void handle_grid_change ();
  void apply_configuration (const configuration &config, std::size_t config_id, grid *solver_grid, int gpu_num);
  void extract (result_extractor **extractors, unsigned int extractors_count);
  bool calculate_next_time_step (result_extractor **extractors, unsigned int extractors_count);
  bool is_gpu_supported () const;

private:
  size_t step = 0;
  double time = 0.0;
  double max_time = 0.0;
  thread_pool threads;
  workspace &solver_workspace;
  std::unique_ptr<solver> solver_context;
};

#endif  // ANYSIM_SIMULATION_MANAGER_H
