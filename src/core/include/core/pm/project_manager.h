//
// Created by egi on 6/2/19.
//

#ifndef ANYSIM_PROJECT_MANAGER_H
#define ANYSIM_PROJECT_MANAGER_H

#include <memory>
#include <vector>

class grid;
class workspace;
class configuration;
class simulation_manager;
class result_extractor;

class project_manager
{
public:
  project_manager ();
  ~project_manager ();

  void initialize (
      std::string project_name_arg,
      std::string solver_arg,
      double max_simulation_time_arg,
      bool use_double_precision_arg);

  bool run ();
  void extract ();
  void update_project ();

  const configuration &get_configuration_scheme () const;
  configuration &get_configuration ();

  const workspace &get_solver_workspace () const;
  const grid &get_grid () const;

  void append_extractor (result_extractor *extractor);

  bool is_double_precision_used () { return use_double_precision; }

  const std::string &get_project_name () { return project_name; }

  void set_gpu_num (int gpu_num_arg)
  {
    if (gpu_num_arg != gpu_num)
    {
      gpu_num = gpu_num_arg;
      version--;
    }
  }

  void set_initializer_script (const std::string &script) { python_initializer = script; version--; }

  bool get_use_gpu () const { return gpu_num >= 0; }

private:
  unsigned int version = 0;
  std::string solver_name;
  std::string project_name;
  bool use_double_precision = true;
  int gpu_num = -1;

  std::string python_initializer;
  std::unique_ptr<grid> solver_grid;
  std::unique_ptr<workspace> solver_workspace;
  std::unique_ptr<configuration> solver_configuration;
  std::unique_ptr<configuration> solver_configuration_scheme;
  std::unique_ptr<simulation_manager> simulation;
  std::vector<result_extractor*> extractors;
};

#endif //ANYSIM_PROJECT_MANAGER_H
