//
// Created by egi on 5/12/19.
//

#ifndef ANYSIM_GUI_SIMULATION_MANAGER_H
#define ANYSIM_GUI_SIMULATION_MANAGER_H

#include <functional>

using compute_action_type = std::function<void(bool)>;
using render_action_type = std::function<void(bool, float*)>;

class gui_simulation_manager
{
public:
  gui_simulation_manager (
    int argc_arg, char *argv_arg[],
    unsigned int nx_arg, unsigned int ny_arg,
    float x_size_arg, float y_size_arg,
    compute_action_type compute_action,
    render_action_type render_action);

  int run ();

private:
  int argc;
  char **argv;
  unsigned int nx, ny;
  float x_size, y_size;
  compute_action_type compute_action;
  render_action_type render_action;
};

#endif //ANYSIM_GUI_SIMULATION_MANAGER_H
