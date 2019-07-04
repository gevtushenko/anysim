//
// Created by egi on 6/16/19.
//

#ifndef ANYSIM_GRID_H
#define ANYSIM_GRID_H

#include <string>
#include <vector>
#include <algorithm>

#include "core/solver/workspace.h"

enum class side_type
{
  left, bottom, right, top
};

enum class boundary_type
{
  mirror, periodic, none
};

inline unsigned int side_to_id (side_type side)
{
  switch (side)
  {
    case side_type::left:   return 0;
    case side_type::bottom: return 1;
    case side_type::right:  return 2;
    case side_type::top:    return 3;
    default:                return 0;
  }

  return 0;
}

inline unsigned int boundary_to_id (boundary_type boundary)
{
  switch (boundary)
  {
    case boundary_type::periodic: return 0;
    case boundary_type::mirror:   return 1;
    case boundary_type::none:     return 2;
    default:                      return 0;
  }
  return 0;
}

constexpr unsigned int unknown_neighbor_id = std::numeric_limits<unsigned int>::max () - 1;
inline bool does_neighbor_exist (unsigned int cell_id) { return cell_id != unknown_neighbor_id; }

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
    // TODO Move to configuration
    boundary_conditions[side_to_id (side_type::left)]   = boundary_to_id (boundary_type::periodic);
    boundary_conditions[side_to_id (side_type::bottom)] = boundary_to_id (boundary_type::mirror);
    boundary_conditions[side_to_id (side_type::right)]  = boundary_to_id (boundary_type::periodic);
    boundary_conditions[side_to_id (side_type::top)]    = boundary_to_id (boundary_type::mirror);

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

  unsigned int get_cell_x (unsigned int cell_id) const { return cell_id % nx; }
  unsigned int get_cell_y (unsigned int cell_id) const { return cell_id / nx; }
  unsigned int get_cell_id (unsigned int x, unsigned int y) const { return y * nx + x; }

  unsigned int get_edges_count (unsigned int /* cell_id */) const { return 4; }

  unsigned int get_neighbor_id (unsigned int cell_id, unsigned int edge_id) const
  {
    const unsigned int x = get_cell_x (cell_id);
    const unsigned int y = get_cell_y (cell_id);

    if (is_edge_left (edge_id))
      return is_cell_on_left_boundary (x) ? get_left_boundary_neighbor (x, y) : get_cell_id (x - 1, y);
    else if (is_edge_bottom (edge_id))
      return is_cell_on_bottom_boundary (y) ? get_bottom_boundary_neighbor (x, y) : get_cell_id (x, y - 1);
    else if (is_edge_right (edge_id))
      return is_cell_on_right_boundary (x) ? get_right_boundary_neighbor (x, y) : get_cell_id (x + 1, y);
    else if (is_edge_top (edge_id))
      return is_cell_on_top_boundary (y) ? get_top_boundary_neighbor (x, y) : get_cell_id (x, y + 1);

    return unknown_neighbor_id;
  }

  float get_cell_volume (unsigned int /* cell_id */) const { return dx * dy; }

  float get_edge_area (unsigned int edge_id) const
  {
    if (is_edge_left (edge_id) || is_edge_right (edge_id))
      return dx;

    if (is_edge_bottom (edge_id) || is_edge_top (edge_id))
      return dy;

    return {};
  }

private:
  bool is_edge_left (unsigned int edge_id) const { return edge_id == side_to_id (side_type::left); }
  bool is_edge_bottom (unsigned int edge_id) const { return edge_id == side_to_id (side_type::bottom); }
  bool is_edge_right (unsigned int edge_id) const { return edge_id == side_to_id (side_type::right); }
  bool is_edge_top (unsigned int edge_id) const { return edge_id == side_to_id (side_type::top); }
  bool is_cell_on_left_boundary (unsigned int x) const { return x == 0; }
  bool is_cell_on_right_boundary (unsigned int x) const { return x == nx - 1; }
  bool is_cell_on_bottom_boundary (unsigned int y) const { return y == 0; }
  bool is_cell_on_top_boundary (unsigned int y) const { return y == ny - 1; }

  unsigned int get_left_boundary_neighbor (unsigned int x, unsigned int y) const
  {
    const auto bc = boundary_conditions[side_to_id (side_type::left)];
    if (bc == boundary_to_id (boundary_type::mirror))
      return get_cell_id (x, y);
    else if (bc == boundary_to_id (boundary_type::periodic))
      return get_cell_id (nx - 1, y);

    /// none
    return unknown_neighbor_id;
  }

  unsigned int get_bottom_boundary_neighbor (unsigned int x, unsigned int y) const
  {
    const auto bc = boundary_conditions[side_to_id (side_type::bottom)];
    if (bc == boundary_to_id (boundary_type::mirror))
      return get_cell_id (x, y);
    else if (bc == boundary_to_id (boundary_type::periodic))
      return get_cell_id (x, ny - 1);

    /// none
    return unknown_neighbor_id;
  }

  unsigned int get_right_boundary_neighbor (unsigned int x, unsigned int y) const
  {
    const auto bc = boundary_conditions[side_to_id (side_type::right)];
    if (bc == boundary_to_id (boundary_type::mirror))
      return get_cell_id (x, y);
    else if (bc == boundary_to_id (boundary_type::periodic))
      return get_cell_id (0, y);

    /// none
    return unknown_neighbor_id;
  }

  unsigned int get_top_boundary_neighbor (unsigned int x, unsigned int y) const
  {
    const auto bc = boundary_conditions[side_to_id (side_type::top)];
    if (bc == boundary_to_id (boundary_type::mirror))
      return get_cell_id (x, y);
    else if (bc == boundary_to_id (boundary_type::periodic))
      return get_cell_id (x, 0);

    /// none
    return unknown_neighbor_id;
  }

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
  unsigned int boundary_conditions[4];
  vertices vertices_2d;
};

#endif //ANYSIM_GRID_H
