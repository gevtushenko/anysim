//
// Created by egi on 6/28/19.
//

#include "io/hdf5/hdf5_writer.h"

#ifdef HDF5_BUILD
#include <hdf5.h>
#endif

class hdf5_writer::hdf5_impl
{
public:
  hdf5_impl (std::string file, project_manager &pm_arg)
    : filename (std::move (file))
    , pm (pm_arg)
  { }

  ~hdf5_impl () { close (); }

  void write_field (
      const void *data,
      const std::string &name,
      bool use_double_precision,
      unsigned int nx,
      unsigned int ny)
  {
    hsize_t dims[3];
    dims[0] = nx;
    dims[1] = ny;

    auto type = use_double_precision ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;

    hid_t dataspace_id = H5Screate_simple(2, dims, nullptr);
    hid_t dataset_id = H5Dcreate2 (file_id, name.c_str (), type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite (dataset_id, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dataset_id);
    H5Sclose (dataspace_id);
  }

  void extract (
    unsigned int thread_id,
    unsigned int threads_count,
    thread_pool &threads)
  {
    if (is_main_thread (thread_id))
    {
      const auto &solver_grid = pm.get_grid ();
      const auto &solver_workspace = pm.get_solver_workspace ();
      const unsigned int nx = solver_grid.nx;
      const unsigned int ny = solver_grid.ny;

      const bool use_double_precision = pm.is_double_precision_used ();

      const std::string time_step_group_name = "/simulation/" + std::to_string (step++);
      hid_t time_step_group_id = H5Gcreate2 (file_id, time_step_group_name.c_str (), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      for (auto &field: solver_grid.get_fields_names())
        write_field (solver_workspace.get (field), time_step_group_name + "/" + field, use_double_precision, nx, ny);

      H5Gclose (time_step_group_id);
    }

    threads.barrier();
  }

  bool open ()
  {
    file_id = H5Fcreate (filename.c_str (), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (check_if_invalid(file_id))
      return true;

    common_group_id = H5Gcreate2 (file_id, "/common", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    simulation_group_id = H5Gcreate2 (file_id, "/simulation", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (!check_if_invalid (common_group_id) && !check_if_invalid (simulation_group_id))
      is_valid = true;

    return is_valid;
  }

  bool close ()
  {
    if (is_valid)
    {
      H5Fclose (file_id);
      H5Gclose (common_group_id);
      H5Gclose (simulation_group_id);

      is_valid = false;
    }
    return false;
  }

private:
  static bool check_if_invalid (const hid_t &id) { return static_cast<int> (id) < 0; }

private:
  bool is_valid = false;

  std::size_t step {};

  hid_t file_id {};
  hid_t common_group_id {};
  hid_t simulation_group_id {};

  std::string filename;
  project_manager &pm;
};

hdf5_writer::hdf5_writer (const std::string &file, project_manager &pm_arg)
  : result_extractor ()
  , implementation (new hdf5_writer::hdf5_impl (file, pm_arg))
{ }

hdf5_writer::~hdf5_writer () = default;

void hdf5_writer::extract (
    unsigned int thread_id,
    unsigned int threads_count,
    thread_pool &threads)
{
  implementation->extract(thread_id, threads_count, threads);
}

bool hdf5_writer::open () { return implementation->open (); }
bool hdf5_writer::close() { return implementation->close (); }
