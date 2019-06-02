//
// Created by egi on 6/2/19.
//

#include "core/pm/project_manager.h"
#include "core/cpu/sources_holder.h"
#include "core/cpu/fdtd_2d.h"
#include "cpp/common_funcs.h"

template <class float_type>
class context_content
{
public:
  context_content () = default;
  ~context_content()
  {
    soft_source.reset ();
    region_initializers.clear ();
    solver.reset ();
  }

  void reset ()
  {
    solver.reset ();
    soft_source.reset ();
    region_initializers.clear ();
  }

  void initialize (
    unsigned int nx_arg, unsigned int ny_arg,
    double calculation_area_width, double calculation_area_height)
  {
    nx = nx_arg;
    ny = nx_arg;

    auto pbc = boundary_condition::periodic;
    solver.reset (new fdtd_2d<float_type> (
      nx, ny,
      calculation_area_width, calculation_area_height,
      pbc, pbc, pbc, pbc));
    soft_source.reset (new sources_holder<float_type> ());
  }

  void initialize_solver ()
  {
    for (auto &initializer: region_initializers)
      solver->initialize_calculation_area (initializer.get ());
  }

  void calculate (bool use_gpu, unsigned int steps)
  {
    solver->calculate (steps, *soft_source, use_gpu);
  }

  void render (bool use_gpu, float *colors)
  {
#ifdef GPU_BUILD
    if (use_gpu)
      {
        fill_colors (nx, ny, solver->get_d_ez (), colors);
      }
    else
#else
      cpp_unreferenced (use_gpu);
#endif
      {
        auto ez = solver->get_ez ();
        for (unsigned int j = 0; j < ny; j++)
          for (unsigned int i = 0; i < nx; i++)
            for (unsigned int k = 0; k < 4; k++)
              fill_vertex_color (ez[j * nx + i], colors + 3 * 4 * (j * nx + i) + 3 * k);
      }
  }

  unsigned int nx = 0, ny = 0;
  std::unique_ptr<fdtd_2d<float_type>> solver;
  std::unique_ptr<sources_holder<float_type>> soft_source;
  std::vector<std::unique_ptr<region_initializer<float_type>>> region_initializers;
};

class calculation_context_interface
{
public:
  virtual ~calculation_context_interface () = default;

  virtual void reset () = 0;
  virtual void initialize (
    unsigned int nx, unsigned int ny,
    double calculation_area_width, double calculation_area_height) = 0;
  virtual void append_source (double frequency, unsigned int mesh_offset) = 0;
  virtual void append_initializer (
    double ez, double hz,
    unsigned int left_bottom_x, unsigned int left_bottom_y,
    unsigned int width, unsigned int height) = 0;
  virtual void initialize_solver () = 0;
  virtual void calculate (bool use_gpu, unsigned int steps) = 0;
  virtual void prepare_gpu () = 0;
  virtual void render (bool use_gpu, float *colors) = 0;

protected:
  bool gpu_was_used = false;
};

class calculation_context_dbl : public calculation_context_interface
{
public:
  calculation_context_dbl () : calculation_context_interface () { }
  ~calculation_context_dbl () override { };

  void reset () override
  {
    #ifdef GPU_BUILD
    if (gpu_was_used)
      context.solver->postprocess_gpu ();
    #endif

    context.reset ();
  }
  void initialize (
    unsigned int nx, unsigned int ny,
    double calculation_area_width, double calculation_area_height) override
  {
    context.initialize (nx, ny, calculation_area_width, calculation_area_height);
  }
  void append_source (double frequency, unsigned int mesh_offset) override
  {
    context.soft_source->append_source (frequency, mesh_offset);
  }
  void append_initializer (
    double ez, double hz,
    unsigned int left_bottom_x, unsigned int left_bottom_y,
    unsigned int width, unsigned int height) override
  {
    context.region_initializers.emplace_back (
      new rectangular_region_initializer<double> (
        context.nx, context.ny,
        left_bottom_x, left_bottom_y,
        width, height,
        ez, hz));
  }
  void initialize_solver () override { context.initialize_solver (); }
  void calculate (bool use_gpu, unsigned int steps) override { context.calculate (use_gpu, steps); }
  virtual void prepare_gpu ()
  {
    #ifdef GPU_BUILD
    gpu_was_used = true;
    context.solver->preprocess_gpu (*context.soft_source);
    #endif
  }
  void render (bool use_gpu, float *colors) override
  {
    context.render (use_gpu, colors);
  }

private:
  context_content<double> context;
};

class calculation_context_flt : public calculation_context_interface
{
public:
  calculation_context_flt () : calculation_context_interface () { }
  ~calculation_context_flt () override { };

  void reset () override
  {
    #ifdef GPU_BUILD
    if (gpu_was_used)
      context.solver->postprocess_gpu ();
    #endif

    context.reset ();
  }
  void initialize (
    unsigned int nx, unsigned int ny,
    double calculation_area_width, double calculation_area_height) override
  {
    context.initialize (nx, ny, calculation_area_width, calculation_area_height);
  }
  void append_source (double frequency, unsigned int mesh_offset) override
  {
    context.soft_source->append_source (frequency, mesh_offset);
  }
  void append_initializer (
    double ez, double hz,
    unsigned int left_bottom_x, unsigned int left_bottom_y,
    unsigned int width, unsigned int height) override
  {
    context.region_initializers.emplace_back (
      new rectangular_region_initializer<float> (
        context.nx, context.ny,
        left_bottom_x, left_bottom_y,
        width, height,
        ez, hz));
  }
  void initialize_solver () override { context.initialize_solver (); }
  void calculate (bool use_gpu, unsigned int steps) override { context.calculate (use_gpu, steps); }
  virtual void prepare_gpu ()
  {
    #ifdef GPU_BUILD
    gpu_was_used = true;
    context.solver->preprocess_gpu (*context.soft_source);
    #endif
  }
  void render (bool use_gpu, float *colors) override
  {
    context.render (use_gpu, colors);
  }

private:
  context_content<float> context;
};

class calculation_context
{
public:
  calculation_context () = delete;
  explicit calculation_context (calculation_context_interface *context_arg)
    : context (context_arg)
  { }

  /// @param width of calculation area in meters
  void set_width (double width)
  {
    calculation_area_width = width;
  }

  /// @param height of calculation area in meters
  void set_height (double height)
  {
    calculation_area_height = height;
  }

  void prepare_solver (
    bool use_gpu,
    unsigned int new_version,
    const std::vector<sources_array> &sources,
    const std::vector<region_initializers_array> &region_initializers)
  {
    if (new_version == version_id)
      return;

    version_id = new_version;
    double max_frequency = 0.1; /// Prevent zero division

    for (auto &source: sources)
      if (source.frequency > max_frequency)
        max_frequency = source.frequency;

    const double lambda_min = C0 / max_frequency;
    dx = lambda_min / cells_per_lambda;
    optimal_nx = static_cast<unsigned int> (std::ceil (calculation_area_width / dx));
    optimal_ny = static_cast<unsigned int> (std::ceil (calculation_area_height / dx));
    dy = calculation_area_height / optimal_ny;

    context->reset ();
    context->initialize (optimal_nx, optimal_ny, calculation_area_width, calculation_area_height);

    for (auto &source: sources)
      {
        const unsigned int mesh_x = std::ceil (source.x / dx);
        const unsigned int mesh_y = std::ceil (source.y / dy);

        context->append_source (source.frequency, optimal_nx * mesh_y + mesh_x);
      }

    if (use_gpu)
      context->prepare_gpu ();

    for (auto &initializer: region_initializers)
      {
        const unsigned int mesh_x = std::ceil (initializer.left_bottom_x / dx);
        const unsigned int mesh_y = std::ceil (initializer.left_bottom_y / dy);
        const unsigned int width = std::ceil (initializer.width / dx);
        const unsigned int height = std::ceil (initializer.height / dy);

        context->append_initializer (initializer.ez, initializer.hz, mesh_x, mesh_y, width, height);
      }

    context->initialize_solver ();
  }

  void calculate (bool use_gpu, unsigned int steps) { context->calculate (use_gpu, steps); }

  unsigned int get_nx () { return optimal_nx; }
  unsigned int get_ny () { return optimal_ny; }

  double get_calculation_area_width () const { return calculation_area_width; }
  double get_calculation_area_height () const { return calculation_area_height; }

  void render (bool use_gpu, float *colors)
  {
    context->render (use_gpu, colors);
  }

protected:
  double dx = 0.1;
  double dy = 0.1;
  unsigned int version_id = 0;
  unsigned int optimal_nx = 10;         /// [cells count]
  unsigned int optimal_ny = 10;         /// [cells count]
  unsigned int cells_per_lambda = 30;   /// [cells count]
  double calculation_area_width  = 5.0; /// [meters]
  double calculation_area_height = 5.0; /// [meters]
  std::unique_ptr<calculation_context_interface> context;
};

static calculation_context_interface *create_context (bool use_double_precision)
{
  if (use_double_precision)
    return new calculation_context_dbl ();
  else
    return new calculation_context_flt ();
}

project_manager::project_manager (bool use_double_precision_arg)
  : use_double_precision (use_double_precision_arg)
  , context (new calculation_context (create_context (use_double_precision)))
{ }

project_manager::~project_manager()
{
  context.reset ();
}

void project_manager::update_version()
{
  version_id++;
}

void project_manager::set_height (double height)
{
  update_version ();
  context->set_height (height);
}

void project_manager::set_width (double width)
{
  update_version ();
  context->set_width (width);
}

void project_manager::append_source (
  double frequency,
  double x,
  double y)
{
  update_version ();
  sources.emplace_back (frequency, x, y);
}

void project_manager::append_initializer (
  double ez_arg,
  double hz_arg,
  double left_bottom_x_arg,
  double left_bottom_y_arg,
  double right_top_x_arg,
  double right_top_y_arg)
{
  initializers.emplace_back (
    ez_arg, hz_arg,
    left_bottom_x_arg, left_bottom_y_arg,
    right_top_x_arg, right_top_y_arg);
}

void project_manager::set_use_gpu(bool use_gpu_arg)
{
  update_version ();
  use_gpu = use_gpu_arg;
}

bool project_manager::get_use_gpu() const
{
  return use_gpu;
}

void project_manager::prepare_simulation()
{
  context->prepare_solver (use_gpu, version_id, sources, initializers);
}

void project_manager::calculate(unsigned int steps)
{
  context->calculate (use_gpu, steps);
}

unsigned int project_manager::get_nx ()
{
  return context->get_nx ();
}

unsigned int project_manager::get_ny ()
{
  return context->get_ny ();
}

void project_manager::render_function (float *colors)
{
  const auto coloring_begin = std::chrono::high_resolution_clock::now ();

  context->render (use_gpu, colors);

  const auto coloring_end = std::chrono::high_resolution_clock::now ();
  const std::chrono::duration<double> duration = coloring_end - coloring_begin;
  std::cout << "Coloring completed in " << duration.count () << "s\n";
}

double project_manager::get_calculation_area_width () const
{
  return context->get_calculation_area_width ();
}

double project_manager::get_calculation_area_height () const
{
  return context->get_calculation_area_height ();
}
