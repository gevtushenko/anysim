//
// Created by egi on 5/12/19.
//

#ifndef ANYSIM_GUI_SIMULATION_MANAGER_H
#define ANYSIM_GUI_SIMULATION_MANAGER_H

class gui_simulation_manager
{
public:
  gui_simulation_manager (int argc_arg, char *argv_arg[]);

  int run ();

private:
  int argc;
  char **argv;
};

#endif //ANYSIM_GUI_SIMULATION_MANAGER_H
