//
// Created by egi on 5/12/19.
//

#ifndef ANYSIM_GUI_APPLICATION_H
#define ANYSIM_GUI_APPLICATION_H

class project_manager;

class gui_application
{
public:
  gui_application (project_manager &pm, int argc_arg, char *argv_arg[]);

  int run ();

private:
  project_manager &pm;
  int argc;
  char **argv;
};

#endif  // ANYSIM_GUI_APPLICATION_H
