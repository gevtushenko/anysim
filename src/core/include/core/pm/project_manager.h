//
// Created by egi on 6/2/19.
//

#ifndef ANYSIM_PROJECT_MANAGER_H
#define ANYSIM_PROJECT_MANAGER_H

#include <memory>
#include <vector>

#include "core/cpu/thread_pool.h"

class sources_array
{
public:
  sources_array (double frequency_arg, double x_arg, double y_arg)
    : frequency (frequency_arg)
    , x (x_arg)
    , y (y_arg)
  { }

  double frequency;
  double x;
  double y;
};

class region_initializers_array
{
public:
  region_initializers_array (
      double ez_arg,
      double hz_arg,
      double left_bottom_x_arg,
      double left_bottom_y_arg,
      double width_arg,
      double height_arg)
    : ez (ez_arg)
    , hz (hz_arg)
    , left_bottom_x (left_bottom_x_arg)
    , left_bottom_y (left_bottom_y_arg)
    , width (width_arg)
    , height (height_arg)
  { }

  double ez;
  double hz;
  double left_bottom_x;
  double left_bottom_y;
  double width;
  double height;
};

class calculation_context;

class project_manager
{
public:
  explicit project_manager (bool use_double_precision);
  ~project_manager ();

  /// @param width of calculation area in meters
  void set_width (double width);

  /// @param height of calculation area in meters
  void set_height (double height);

  void set_cells_per_lambda (unsigned int cells_per_lambda);

  void set_use_gpu (bool use_gpu_arg);
  bool get_use_gpu () const;

  void append_source (double frequency, double x, double y);

  void prepare_simulation ();
  void calculate (unsigned int steps);
  void render_function (float *colors);

  void append_initializer (
      double ez_arg,
      double hz_arg,
      double left_bottom_x_arg,
      double left_bottom_y_arg,
      double right_top_x_arg,
      double right_top_y_arg);

  unsigned int get_nx ();
  unsigned int get_ny ();

  double get_calculation_area_width () const;
  double get_calculation_area_height () const;

private:
  void update_version ();

private:
  thread_pool threads;
  bool use_gpu = false;
  unsigned int version_id = 0;
  bool use_double_precision = true;
  std::unique_ptr<calculation_context> context;
  std::vector<sources_array> sources;
  std::vector<region_initializers_array> initializers;
};

#endif //ANYSIM_PROJECT_MANAGER_H
