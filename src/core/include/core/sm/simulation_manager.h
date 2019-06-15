//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_SIMULATION_MANAGER_H
#define ANYSIM_SIMULATION_MANAGER_H

#include <string>
#include <memory>

#include "core/cpu/thread_pool.h"

class solver;
class configuration;

class simulation_manager
{
public:
  simulation_manager (const std::string &solver_arg, bool use_double_precision_arg);
  void fill_configuration_scheme (configuration &config);

private:
  thread_pool threads;
  std::unique_ptr<solver> solver_context;
};

#endif  // ANYSIM_SIMULATION_MANAGER_H
