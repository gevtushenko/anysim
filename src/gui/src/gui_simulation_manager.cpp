//
// Created by egi on 5/12/19.
//

#include "gui_simulation_manager.h"
#include "main_window.h"

#include <QApplication>

gui_simulation_manager::gui_simulation_manager(
    int argc_arg,
    char **argv_arg)
  : argc (argc_arg)
  , argv (argv_arg)
{ }

int gui_simulation_manager::run()
{
  QApplication app (argc, argv);

  main_window window;
  window.resize (QSize (1000, 800));
  window.show ();

  return app.exec ();
}

