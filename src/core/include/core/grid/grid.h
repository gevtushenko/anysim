//
// Created by egi on 6/16/19.
//

#ifndef ANYSIM_GRID_H
#define ANYSIM_GRID_H

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "core/common/common_defs.h"
#include "core/solver/workspace.h"
#include "core/grid/geometry.h"

enum class side_type
{
  left, bottom, right, top
};

enum class boundary_type
{
  mirror, periodic, none
};

inline CPU_GPU unsigned int side_to_id (side_type side)
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

inline CPU_GPU unsigned int boundary_to_id (boundary_type boundary)
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
inline CPU_GPU bool does_neighbor_exist (unsigned int cell_id) { return cell_id != unknown_neighbor_id; }

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

inline CPU_GPU bool is_edge_left (unsigned int edge_id) { return edge_id == side_to_id (side_type::left); }
inline CPU_GPU bool is_edge_bottom (unsigned int edge_id) { return edge_id == side_to_id (side_type::bottom); }
inline CPU_GPU bool is_edge_right (unsigned int edge_id) { return edge_id == side_to_id (side_type::right); }
inline CPU_GPU bool is_edge_top (unsigned int edge_id) { return edge_id == side_to_id (side_type::top); }

class grid_topology
{
public:
  void initialize_for_structured_uniform_grid (
    unsigned int nx_arg,
    unsigned int ny_arg,
    unsigned int bc_left,
    unsigned int bc_bottom,
    unsigned int bc_right,
    unsigned int bc_top)
  {
    boundary_conditions[side_to_id (side_type::left)] = bc_left;
    boundary_conditions[side_to_id (side_type::bottom)] = bc_bottom;
    boundary_conditions[side_to_id (side_type::right)] = bc_right;
    boundary_conditions[side_to_id (side_type::top)] = bc_top;

    nx = nx_arg;
    ny = ny_arg;
    n_cells = nx * ny;

    complex_topology = nullptr;
    complex_topology_mapping = nullptr;
  }

  CPU_GPU unsigned int get_cells_count () const { return n_cells; }
  CPU_GPU unsigned int get_cell_x (unsigned int cell_id) const { return cell_id % nx; }
  CPU_GPU unsigned int get_cell_y (unsigned int cell_id) const { return cell_id / nx; }
  CPU_GPU unsigned int get_cell_id (unsigned int x, unsigned int y) const { return y * nx + x; }

  CPU_GPU unsigned int get_edges_count (unsigned int /* cell_id */) const { return 4; }

  CPU_GPU unsigned int get_neighbor_id (unsigned int cell_id, unsigned int edge_id) const
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

private:
  CPU_GPU bool is_complex () const
  {
    return complex_topology && complex_topology_mapping;
  }

  CPU_GPU bool is_cell_on_left_boundary (unsigned int x) const { return x == 0; }
  CPU_GPU bool is_cell_on_right_boundary (unsigned int x) const { return x == nx - 1; }
  CPU_GPU bool is_cell_on_bottom_boundary (unsigned int y) const { return y == 0; }
  CPU_GPU bool is_cell_on_top_boundary (unsigned int y) const { return y == ny - 1; }

  CPU_GPU unsigned int get_left_boundary_neighbor (unsigned int x, unsigned int y) const
  {
    const auto bc = boundary_conditions[side_to_id (side_type::left)];
    if (bc == boundary_to_id (boundary_type::mirror))
      return get_cell_id (x, y);
    else if (bc == boundary_to_id (boundary_type::periodic))
      return get_cell_id (nx - 1, y);

    /// none
    return unknown_neighbor_id;
  }

  CPU_GPU unsigned int get_bottom_boundary_neighbor (unsigned int x, unsigned int y) const
  {
    const auto bc = boundary_conditions[side_to_id (side_type::bottom)];
    if (bc == boundary_to_id (boundary_type::mirror))
      return get_cell_id (x, y);
    else if (bc == boundary_to_id (boundary_type::periodic))
      return get_cell_id (x, ny - 1);

    /// none
    return unknown_neighbor_id;
  }

  CPU_GPU unsigned int get_right_boundary_neighbor (unsigned int x, unsigned int y) const
  {
    const auto bc = boundary_conditions[side_to_id (side_type::right)];
    if (bc == boundary_to_id (boundary_type::mirror))
      return get_cell_id (x, y);
    else if (bc == boundary_to_id (boundary_type::periodic))
      return get_cell_id (0, y);

    /// none
    return unknown_neighbor_id;
  }

  CPU_GPU unsigned int get_top_boundary_neighbor (unsigned int x, unsigned int y) const
  {
    const auto bc = boundary_conditions[side_to_id (side_type::top)];
    if (bc == boundary_to_id (boundary_type::mirror))
      return get_cell_id (x, y);
    else if (bc == boundary_to_id (boundary_type::periodic))
      return get_cell_id (x, 0);

    /// none
    return unknown_neighbor_id;
  }

private:
  unsigned int n_cells;
  unsigned int nx, ny;
  unsigned int *complex_topology;
  unsigned int *complex_topology_mapping;
  unsigned int boundary_conditions[4];
};

class grid_geometry
{
public:
  void initialize_for_structured_uniform_grid (
    unsigned int nx_arg, unsigned int ny_arg,
    float dx_arg, float dy_arg)
  {
    nx = nx_arg;
    ny = ny_arg;
    dx = dx_arg;
    dy = dy_arg;
  }

  CPU_GPU float get_cell_volume (unsigned int /* cell_id */) const { return dx * dy; }

  CPU_GPU float get_edge_area (unsigned int /*cell_id*/, unsigned int edge_id) const
  {
    if (is_edge_left (edge_id) || is_edge_right (edge_id))
      return dx;

    if (is_edge_bottom (edge_id) || is_edge_top (edge_id))
      return dy;

    return {};
  }

  CPU_GPU float get_cell_center_x (unsigned int cell_id) const
  {
    const unsigned int x = cell_id % nx;
    return x * dx + dx / 2.0;
  }

  CPU_GPU float get_cell_center_y (unsigned int cell_id) const
  {
    const unsigned int y = cell_id / nx;
    return y * dy + dy / 2.0;
  }

  CPU_GPU float get_normal_x (unsigned int /* cell_id */, unsigned int edge_id) const
  {
    if (is_edge_left (edge_id))
      return -1.0;
    else if (is_edge_bottom (edge_id) || is_edge_top (edge_id))
      return 0.0;
    else if (is_edge_right (edge_id))
      return 1.0;

    return 0.0;
  }

  CPU_GPU float get_normal_y (unsigned int /* cell_id */, unsigned int edge_id) const
  {
    if (is_edge_left (edge_id) || is_edge_right (edge_id))
      return 0.0;
    else if (is_edge_bottom (edge_id))
      return -1.0;
    else if (is_edge_top (edge_id))
      return 1.0;

    return 0.0;
  }

  CPU_GPU float get_distance_between_cells_x (unsigned int first_cell, unsigned int second_cell) const
  {
    return std::abs (get_cell_center_x (first_cell) - get_cell_center_x (second_cell));
  }

  CPU_GPU float get_distance_between_cells_y (unsigned int first_cell, unsigned int second_cell) const
  {
    return std::abs (get_cell_center_y (first_cell) - get_cell_center_y (second_cell));
  }

  CPU_GPU unsigned int get_cell_id_by_coordinates (float x, float y) const
  {
    const unsigned int grid_x = std::ceil (x / dx);
    const unsigned int grid_y = std::ceil (y / dy);
    return nx * grid_y + grid_x;
  }

private:
  float dx, dy;
  unsigned int nx, ny;
};

static_assert(std::is_pod<grid_topology>::value, "Class grid_topology has to be POD");
static_assert(std::is_pod<grid_geometry>::value, "Class grid_geometry has to be POD");

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
  , gl_representation (geometry_element_type::quad, nx * ny)
  {
    vertices_2d.allocate (nx * ny * vertices_per_cell, coordinates_per_vertex);
    float *v = vertices_2d.get_vertices ();

    const auto dxf = static_cast<float> (dx);
    const auto dyf = static_cast<float> (dy);

    for (unsigned int j = 0; j < ny; j++)
      {
        const auto fj = static_cast<float> (j);

        for (unsigned int i = 0; i < nx; i++)
          {
            const auto fi = static_cast<float> (i);
            const unsigned int vert_offset = vertex_data_per_element * (j * nx + i);

            v[vert_offset + 0] = dxf * (fi + 0); v[vert_offset + 1] = dyf * (fj + 1);
            v[vert_offset + 2] = dxf * (fi + 0); v[vert_offset + 3] = dyf * (fj + 0);
            v[vert_offset + 4] = dxf * (fi + 1); v[vert_offset + 5] = dyf * (fj + 0);
            v[vert_offset + 6] = dxf * (fi + 1); v[vert_offset + 7] = dyf * (fj + 1);

            gl_representation.append_pixel (point (dxf * fi, dyf * fj), sizes_set (dxf, dyf));
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

  [[nodiscard]] const std::vector<std::string> &get_fields_names () const { return fields_names; }

  std::size_t get_cells_number () const { return nx * ny; }

  const float *get_vertices_data () const { return vertices_2d.get_vertices (); }

  grid_geometry gen_geometry_wrapper () const
  {
    grid_geometry geometry;
    geometry.initialize_for_structured_uniform_grid (nx, ny, dx, dy);
    return geometry;
  }

  grid_topology gen_topology_wrapper () const
  {
    grid_topology topology;
    topology.initialize_for_structured_uniform_grid (
      nx, ny,
      boundary_to_id (boundary_type::periodic),
      boundary_to_id (boundary_type::periodic),
      boundary_to_id (boundary_type::periodic),
      boundary_to_id (boundary_type::periodic));
    return topology;
  }

  float get_bounding_box_width () const { return width; }
  float get_bounding_box_height () const { return height; }

  const geometry_representation &get_gl_representation () const;

public:
  const std::size_t vertices_per_cell = 4;
  const std::size_t coordinates_per_vertex = 2;
  const std::size_t vertex_data_per_element = vertices_per_cell * coordinates_per_vertex;

private:
  const unsigned int nx;
  const unsigned int ny;
  const unsigned int size;
  const double width;
  const double height;
  const double dx;
  const double dy;

  workspace &solver_workspace;
  std::vector<std::string> fields_names;

  vertices vertices_2d;
  geometry_representation gl_representation;
};

#endif //ANYSIM_GRID_H
