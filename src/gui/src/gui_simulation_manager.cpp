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
    std::function<void(float *)> render_action_arg)
    : argc (argc_arg)
    , argv (argv_arg)
    , nx (nx_arg)
    , ny (ny_arg)
    , x_size (x_size_arg)
    , y_size (y_size_arg)
    , render_action (render_action_arg)
{
}

int gui_simulation_manager::run()
{
  QApplication app (argc, argv);

  main_window window(
      nx, ny,
      static_cast<float>(x_size), static_cast<float>(y_size), render_action);
  window.resize (QSize (800, 800));
  window.show ();

  return app.exec ();
}

