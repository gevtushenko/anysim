//
// Created by egi on 5/12/19.
//

#ifndef ANYSIM_GUI_SIMULATION_MANAGER_H
#define ANYSIM_GUI_SIMULATION_MANAGER_H

#include <functional>

class gui_simulation_manager
{
public:
  gui_simulation_manager (
    int argc_arg, char *argv_arg[],
    unsigned int nx_arg, unsigned int ny_arg,
    float x_size_arg, float y_size_arg,
    std::function<void(void)> compute_action_arg,
    std::function<void(float *)> render_action_arg);

  int run ();

private:
  int argc;
  char **argv;
  unsigned int nx, ny;
  float x_size, y_size;
  std::function<void()> compute_action;
  std::function<void(float *)> render_action;
};

#endif //ANYSIM_GUI_SIMULATION_MANAGER_H
