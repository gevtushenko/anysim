//
// Created by egi on 6/16/19.
//

#ifndef ANYSIM_GRID_H
#define ANYSIM_GRID_H

#include <string>
#include <vector>
#include <algorithm>

#include "core/solver/workspace.h"

class vertices
{
public:
  void allocate (std::size_t vertices_number_arg, std::size_t coords_per_vertex)
  {
    vertices_number = vertices_number_arg;

    if (vertices_number > allocated_vertices_number)
      {
        v.reset ();
        v.reset (new float[vertices_number_arg * coords_per_vertex]);

        allocated_vertices_number = vertices_number;
      }
  }

  float *get_vertices () { return v.get (); }
  const float *get_vertices () const { return v.get (); }

private:
  std::size_t vertices_number = 0;
  std::size_t allocated_vertices_number = 0;

  std::unique_ptr<float[]> v;
};

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

    vertices_2d.allocate (nx * ny * vertices_per_cell, coordinates_per_vertex);
    float *v = vertices_2d.get_vertices ();

    for (unsigned int j = 0; j < ny; j++)
      {
        for (unsigned int i = 0; i < nx; i++)
          {
            const unsigned int vert_offset = vertex_data_per_element * (j * nx + i);

            v[vert_offset + 0] = dx * (i + 0); v[vert_offset + 1] = dy * (j + 1);
            v[vert_offset + 2] = dx * (i + 0); v[vert_offset + 3] = dy * (j + 0);
            v[vert_offset + 4] = dx * (i + 1); v[vert_offset + 5] = dy * (j + 0);
            v[vert_offset + 6] = dx * (i + 1); v[vert_offset + 7] = dy * (j + 1);
          }
      }
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

  std::size_t get_cells_number () const { return nx * ny; }

  const float *get_vertices_data () const { return vertices_2d.get_vertices (); }

public:
  const unsigned int nx;
  const unsigned int ny;
  const unsigned int size;
  const double width;
  const double height;
  const double dx;
  const double dy;

  const std::size_t vertices_per_cell = 4;
  const std::size_t coordinates_per_vertex = 2;
  const std::size_t vertex_data_per_element = vertices_per_cell * coordinates_per_vertex;

private:
  workspace &solver_workspace;
  std::vector<std::string> fields_names;
  vertices vertices_2d;
};

#endif //ANYSIM_GRID_H
