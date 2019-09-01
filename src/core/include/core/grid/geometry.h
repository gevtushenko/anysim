//
// Created by egi on 9/1/19.
//

#ifndef ANYSIM_GEOMETRY_H
#define ANYSIM_GEOMETRY_H

#include <vector>
#include <array>

enum class geometry_element_type : int
{
  vertex, line, quad,          /// 2d
  tetra, hexahedron, pyramid   /// 3d
};

struct sizes_set
{
  explicit sizes_set (float w=0.0, float h=0.0, float d=0.0)
    : data ({w, h, d})
    , width (data[0])
    , height (data[1])
    , depth (data[2])
  { }

  static constexpr int size = 3;
  std::array<float, size> data;

  float &width;
  float &height;
  float &depth;
};

struct point
{
  explicit point (float xv=0.0, float yv=0.0, float zv=0.0)
    : data ({xv, yv, zv})
    , x (data[0])
    , y (data[1])
    , z (data[2])
  { }

  static constexpr int size = 3;
  std::array<float, size> data;

  float &x;
  float &y;
  float &z;
};

inline point operator+ (const point &lhs, const sizes_set &rhs)
{
  point result;

  for (unsigned int idx = 0; idx < point::size; idx++)
    result.data[idx] = lhs.data[idx] + rhs.data[idx];

  return result;
}

inline point operator- (const point &lhs, const sizes_set &rhs)
{
  point result;

  for (unsigned int idx = 0; idx < point::size; idx++)
    result.data[idx] = lhs.data[idx] - rhs.data[idx];

  return result;
}

struct boundary_box
{
  point left_bottom_corner;
  point right_top_corner;

  [[nodiscard]] float width () const { return right_top_corner.x - left_bottom_corner.x; }
  [[nodiscard]] float height () const { return right_top_corner.y - left_bottom_corner.y; }
  [[nodiscard]] float depth () const { return right_top_corner.z - left_bottom_corner.z; }
};

/**
 * This class is used to separate geometry structures from their GL representation.
 */
class geometry_representation
{
public:
  explicit geometry_representation (
      geometry_element_type element_type_arg,
      size_t preserve_elements_count=0);

  [[nodiscard]] size_t get_elements_count () const;
  [[nodiscard]] size_t get_vertices_count () const;
  [[nodiscard]] boundary_box get_boundary_box () const;

  [[nodiscard]] const float *data () const { return vertices.data (); }
  [[nodiscard]] size_t size () const { return vertices.size (); }
  [[nodiscard]] unsigned int get_vertices_per_element () const { return vertices_per_element; }
  [[nodiscard]] geometry_element_type get_element_type () const { return elements_type; }

  void append_pixel (const point &left_bottom_corner, const sizes_set &sizes);
  void append_vertex (const point &v);

private:
  void update_boundary (const point &corner);

private:
  static constexpr unsigned int dimensions = 3;
  unsigned int vertices_per_element;
  geometry_element_type elements_type;
  boundary_box boundary;

  std::vector<float> vertices;
};

#endif  // ANYSIM_GEOMETRY_H
