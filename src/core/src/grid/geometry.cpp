//
// Created by egi on 9/1/19.
//

#include "core/grid/geometry.h"

static unsigned int element_type_vertices_count (geometry_element_type element_type)
{
  switch (element_type)
  {
    case geometry_element_type::vertex:     return 1;
    case geometry_element_type::line:       return 2;
    case geometry_element_type::quad:       return 4;
    case geometry_element_type::tetra:      return 4;
    case geometry_element_type::hexahedron: return 8;
    case geometry_element_type::pyramid:    return 5;
  }

  return 0;
}

geometry_representation::geometry_representation (
    geometry_element_type element_type_arg,
    size_t preserve_elements_count)
  : vertices_per_element (element_type_vertices_count (element_type_arg))
  , elements_type (element_type_arg)
{
  vertices.reserve (preserve_elements_count * vertices_per_element  * dimensions);
}

void geometry_representation::append_vertex (const point &v)
{
  vertices.push_back (v.x);
  vertices.push_back (v.y);
  vertices.push_back (v.z);
}

void geometry_representation::append_pixel (const point &left_bottom_corner, const sizes_set &sizes)
{
  update_boundary (left_bottom_corner);
  update_boundary (left_bottom_corner + sizes);

  append_vertex (left_bottom_corner + sizes_set (0, sizes.height));
  append_vertex (left_bottom_corner);
  append_vertex (left_bottom_corner + sizes_set (sizes.width));
  append_vertex (left_bottom_corner + sizes_set (sizes.width, sizes.height));
}

void geometry_representation::update_boundary (const point &corner)
{
  for (unsigned int idx = 0; idx < point::size; idx++)
  {
    if (corner.data[idx] < boundary.left_bottom_corner.data[idx])
      boundary.left_bottom_corner.data[idx] = corner.data[idx];

    if (corner.data[idx] > boundary.right_top_corner.data[idx])
      boundary.right_top_corner.data[idx] = corner.data[idx];
  }
}

size_t geometry_representation::get_vertices_count () const
{
  return vertices.size () / dimensions;
}

size_t geometry_representation::get_elements_count () const
{
  return get_vertices_count () / vertices_per_element;
}

boundary_box geometry_representation::get_boundary_box () const
{
  return boundary;
}

