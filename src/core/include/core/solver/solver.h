//
// Created by egi on 6/15/19.
//

#ifndef ANYSIM_SOLVER_H
#define ANYSIM_SOLVER_H

class configuration;

class solver
{
public:
  virtual ~solver () = default;

  /// Computes one step
  virtual void solve (unsigned int step, unsigned int thread_id, unsigned int total_threads) = 0;
  virtual void apply_configuration (const configuration &config) = 0;
  virtual void fill_configuration_scheme (configuration &config) = 0;
};

#endif //ANYSIM_SOLVER_H
