//
// Created by egi on 6/2/19.
//

#include "core/pm/project_manager.h"
#include "core/config/configuration.h"
#include "core/sm/simulation_manager.h"
#include "core/sm/result_extractor.h"
#include "core/solver/solver.h"
#include "core/grid/grid.h"
#include "core/cpu/euler_2d.h"
#include "core/solver/workspace.h"

#include <iostream>

#ifdef PYTHON_BUILD
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace py::literals;
#endif

project_manager::project_manager ()
  : solver_workspace (new workspace ())
{ }

project_manager::~project_manager () = default;

void project_manager::initialize (
    std::string project_name_arg,
    std::string solver_arg,
    double max_simulation_time_arg,
    bool use_double_precision_arg)
{
  solver_name = std::move (solver_arg);
  project_name = std::move (project_name_arg);
  use_double_precision = use_double_precision_arg;

  solver_configuration = std::make_unique<configuration> ();
  solver_configuration_scheme = std::make_unique<configuration> ();
  simulation = std::make_unique<simulation_manager> (
      solver_name,
      max_simulation_time_arg,
      use_double_precision,
      *solver_workspace);

  auto &scheme = *solver_configuration_scheme;
  const auto grid_id = scheme.create_group (scheme.get_root (), "grid");
  scheme.create_node (grid_id, "nx", 10 /* default value */);
  scheme.create_node (grid_id, "ny", 10 /* default value */);
  scheme.create_node (grid_id, "width",  1.0 /* default value */);
  scheme.create_node (grid_id, "height", 1.0 /* default value */);
  const auto solver_id = scheme.create_group (scheme.get_root (), "solver");
  simulation->fill_configuration_scheme (scheme, solver_id);

  // configuration_node grid_part ("grid");
  // grid_part.append_node ("nx", 10 /* default value */);
  // grid_part.append_node ("ny", 10 /* default value */);
  // grid_part.append_node ("width",  1.0 /* default value */);
  // grid_part.append_node ("height", 1.0 /* default value */);
  // solver_configuration_scheme->get_root ().append_node (grid_part);

  // auto solver_part = solver_configuration_scheme->get_root ().append_and_get_group ("solver");
  // simulation->fill_configuration_scheme (*solver_part);
  // solver_configuration_scheme->get_root ().print ();
}

const configuration& project_manager::get_configuration_scheme () const
{
  return *solver_configuration_scheme;
}

configuration &project_manager::get_configuration ()
{
  return *solver_configuration;
}

#ifdef PYTHON_BUILD
PYBIND11_EMBEDDED_MODULE(anysim_py, m) {
  // `m` is a `py::module` which is used to bind functions and classes
  m.def("add", [](int i, int j) {
    return i + j;
  });
  py::class_<grid_topology>(m, "grid_topology")
    .def(py::init<>())
    .def("get_cells_count", &grid_topology::get_cells_count);
  py::class_<grid_geometry>(m, "grid_geometry")
    .def(py::init<>())
    .def("get_cell_center_x", &grid_geometry::get_cell_center_x)
    .def("get_cell_center_y", &grid_geometry::get_cell_center_y);
}

template<class data_type>
py::array_t<data_type> create_py_array (size_t n, void *data_ptr)
{
  /// Specify 'owner' for data to prevent numpy from copying arrays
  py::capsule free_when_done(data_ptr, [](void *f) { (void) f; });
  return py::array_t<data_type> (n, reinterpret_cast<data_type *> (data_ptr), free_when_done);
}
#endif

void project_manager::update_project ()
{
  const auto &config = *solver_configuration;
  if (version != config.get_version ())
  {
    version = config.get_version ();

    auto grid_node_id = config.children_for (config.get_root ()).front ();
    auto grid_params = config.children_for (grid_node_id);
    const unsigned int nx = config.get_node_value (grid_params[0]);
    const unsigned int ny = config.get_node_value (grid_params[1]);

    const double width  = config.get_node_value (grid_params[2]);
    const double height = config.get_node_value (grid_params[3]);

    if (get_use_gpu ())
      if (!simulation->is_gpu_supported ())
        gpu_num = -1;

    solver_grid = std::make_unique<grid> (*solver_workspace, nx, ny, width, height);
    simulation->apply_configuration (config, config.children_for (config.get_root ()).at (1), solver_grid.get (), gpu_num);

#ifdef PYTHON_BUILD
    if (!python_initializer.empty ())
      {
        auto topology = solver_grid->gen_topology_wrapper ();
        auto geometry = solver_grid->gen_geometry_wrapper ();

        py::scoped_interpreter guard{};
        auto anysim_py_module = py::module::import ("anysim_py");
        anysim_py_module.attr ("topology") = py::cast (topology);
        anysim_py_module.attr ("geometry") = py::cast (geometry);

        py::dict kwargs;
        for (auto &field: solver_grid->get_fields_names ())
          {
            if (use_double_precision)
              kwargs[field.c_str ()] = create_py_array<double> (topology.get_cells_count (), solver_workspace->get (field));
            else
              kwargs[field.c_str ()] = create_py_array<float> (topology.get_cells_count (), solver_workspace->get (field));
          }

        anysim_py_module.attr ("fields") = kwargs;
        py::exec(python_initializer);
        simulation->handle_grid_change ();
      }
#endif
  }
}

bool project_manager::run ()
{
  update_project ();
  return simulation->calculate_next_time_step (extractors.data (), extractors.size ());
}

void project_manager::extract ()
{
  update_project ();
  simulation->extract (extractors.data (), extractors.size ());
}

const workspace &project_manager::get_solver_workspace () const
{
  return *solver_workspace;
}

const grid &project_manager::get_grid () const
{
  return *solver_grid;
}

void project_manager::append_extractor (result_extractor *extractor)
{
  extractors.push_back (extractor);
}
