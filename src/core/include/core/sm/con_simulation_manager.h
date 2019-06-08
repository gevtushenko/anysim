//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_CON_SIMULATION_MANAGER_H
#define ANYSIM_CON_SIMULATION_MANAGER_H

class project_manager;

class simulation_manager
{
public:
  virtual ~simulation_manager () = default;
  virtual int run () = 0;
};

class con_simulation_manager : public simulation_manager
{
public:
  con_simulation_manager () = delete;
  explicit con_simulation_manager (project_manager &pm);

  int run () override;

private:
  project_manager &pm;
};

#endif //ANYSIM_CON_SIMULATION_MANAGER_H
