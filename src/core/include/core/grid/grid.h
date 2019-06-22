//
// Created by egi on 6/16/19.
//

#ifndef ANYSIM_GRID_H
#define ANYSIM_GRID_H

#include <string>
#include <vector>
#include <algorithm>

#include "core/solver/workspace.h"

class grid
{
public:
  grid (
    workspace &workspace_arg,
    unsigned int nx_arg,
    unsigned int ny_arg,
    double width_arg,
    double height_arg)
  : nx (nx_arg)
  , ny (ny_arg)
  , size (nx * ny)
  , width (width_arg)
  , height (height_arg)
  , dx (width / nx)
  , dy (height / ny)
  , solver_workspace (workspace_arg)
  {

  }

  template <class field_type>
  bool create_field (const std::string &field_name, memory_holder_type holder, unsigned int layouts)
  {
    if (solver_workspace.allocate (field_name, holder, size * sizeof (field_type), layouts))
      return true;
    if (std::find (fields_names.begin (), fields_names.end (), field_name) == fields_names.end ())
      fields_names.push_back (field_name);
    return false;
  }

  const std::vector<std::string> &get_fields_names () const { return fields_names; }

public:
  const unsigned int nx;
  const unsigned int ny;
  const unsigned int size;
  const double width;
  const double height;
  const double dx;
  const double dy;

private:
  workspace &solver_workspace;
  std::vector<std::string> fields_names;
};

#endif //ANYSIM_GRID_H
