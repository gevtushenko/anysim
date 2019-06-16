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
class result_extractor;
class configuration_node;

class simulation_manager
{
public:
  simulation_manager (
      const std::string &solver_arg,
      bool use_double_precision_arg,
      workspace &workspace_arg);
  void fill_configuration_scheme (configuration_node &solver_scheme);
  void apply_configuration (const configuration_node &solver_config, grid &solver_grid);
  bool calculate_next_time_step ();
  void append_extractor (result_extractor *extractor);

private:
  size_t step = 0;
  thread_pool threads;
  workspace &solver_workspace;
  std::unique_ptr<solver> solver_context;
  std::vector<result_extractor*> extractors;
};

#endif  // ANYSIM_SIMULATION_MANAGER_H
