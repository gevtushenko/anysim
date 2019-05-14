//
// Created by egi on 5/12/19.
//

#include <QApplication>

#include "gui_simulation_manager.h"
#include "main_window.h"


gui_simulation_manager::gui_simulation_manager(
    int argc_arg,
    char **argv_arg,
    unsigned int nx_arg,
    unsigned int ny_arg,
    float x_size_arg,
    float y_size_arg,
    compute_action_type compute_action_arg,
    render_action_type render_action_arg)
    : argc (argc_arg)
    , argv (argv_arg)
    , nx (nx_arg)
    , ny (ny_arg)
    , x_size (x_size_arg)
    , y_size (y_size_arg)
    , compute_action (compute_action_arg)
    , render_action (render_action_arg)
{
}

int gui_simulation_manager::run()
{
  QApplication app (argc, argv);

  main_window window(
      nx, ny,
      static_cast<float>(x_size), static_cast<float>(y_size), compute_action, render_action);
  window.resize (QSize (800, 800));
  window.show ();

  return app.exec ();
}

