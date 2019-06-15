//
// Created by egi on 6/2/19.
//

#ifndef ANYSIM_PROJECT_MANAGER_H
#define ANYSIM_PROJECT_MANAGER_H

#include <memory>

class configuration;
class simulation_manager;

class project_manager
{
public:
  project_manager ();
  ~project_manager ();

  void initialize (
      std::string project_name_arg,
      std::string solver_arg,
      bool use_double_precision_arg);

  const configuration &get_configuration_scheme () const;
  configuration &get_configuration ();

private:
  std::string solver_name;
  std::string project_name;
  bool use_double_precision = true;

  std::unique_ptr<configuration> solver_configuration;
  std::unique_ptr<configuration> solver_configuration_scheme;
  std::unique_ptr<simulation_manager> simulation;
};

#endif //ANYSIM_PROJECT_MANAGER_H
