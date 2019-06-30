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

#if HDF5_BUILD
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
#endif

  void extract (
    unsigned int thread_id,
    unsigned int threads_count,
    thread_pool &threads)
  {
#if HDF5_BUILD
    if (is_main_thread (thread_id))
    {
      const auto &solver_grid = pm.get_grid ();
      const auto &solver_workspace = pm.get_solver_workspace ();
      const unsigned int nx = solver_grid.nx;
      const unsigned int ny = solver_grid.ny;

      if (step == 0)
      {
        const double width = solver_grid.width;
        const double height = solver_grid.height;
        const double dx = width / nx;
        const double dy = height / ny;

        std::unique_ptr<float[]> x (new float[(nx + 1) * (ny + 1)]);
        std::unique_ptr<float[]> y (new float[(nx + 1) * (ny + 1)]);

        for (unsigned int j = 0; j < ny + 1; j++)
        {
          for (unsigned int i = 0; i < nx + 1; i++)
          {
            const unsigned int idx = j * (nx + 1) + i;

            x[idx] = dx * i;
            y[idx] = dy * j;
          }
        }

        const std::string x_group_name = "/common/x";
        const std::string y_group_name = "/common/y";

        write_field (x.get (), x_group_name, false, nx + 1, ny + 1);
        write_field (y.get (), y_group_name, false, nx + 1, ny + 1);

        write_xdmf_xml_head (nx, ny);
      }

      const bool use_double_precision = pm.is_double_precision_used ();
      write_xdmf_xml_body (nx, ny, use_double_precision, solver_grid.get_fields_names());

      const std::string time_step_group_name = "/simulation/" + std::to_string (step++);
      hid_t time_step_group_id = H5Gcreate2 (file_id, time_step_group_name.c_str (), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      for (auto &field: solver_grid.get_fields_names())
        write_field (solver_workspace.get_host_copy (field), time_step_group_name + "/" + field, use_double_precision, nx, ny);

      H5Gclose (time_step_group_id);
    }

    threads.barrier();
#endif
  }

  bool open ()
  {
#if HDF5_BUILD
    std::string hdf5_filename = filename + ".h5";
    file_id = H5Fcreate (hdf5_filename.c_str (), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (check_if_invalid(file_id))
      return true;

    common_group_id = H5Gcreate2 (file_id, "/common", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    simulation_group_id = H5Gcreate2 (file_id, "/simulation", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (!check_if_invalid (common_group_id) && !check_if_invalid (simulation_group_id))
      is_valid = true;

#endif
    return is_valid;
  }

  bool close ()
  {
#if HDF5_BUILD
    if (is_valid)
    {
      H5Fclose (file_id);
      H5Gclose (common_group_id);
      H5Gclose (simulation_group_id);

      write_xdmf_xml_tail ();

      is_valid = false;
    }
#endif
    return false;
  }

private:
#if HDF5_BUILD
  static bool check_if_invalid (const hid_t &id) { return static_cast<int> (id) < 0; }

  void write_xdmf_xml_head (unsigned int nx, unsigned int ny)
  {
    std::string hdf_filename = filename + ".h5";
    std::string xmf_filename = filename + ".xmf";
    xmf = fopen(xmf_filename.c_str (), "w");
    fprintf (xmf, "<?xml version=\"1.0\" ?>\n");
    fprintf (xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf (xmf, "<Xdmf Version=\"2.0\">\n");
    fprintf (xmf, " <Domain>\n");
    fprintf (xmf, "   <Topology TopologyType=\"2DSMesh\" NumberOfElements=\"%u %u\"/>\n", ny + 1, nx + 1);
    fprintf (xmf, "   <Geometry GeometryType=\"X_Y\">\n");
    fprintf (xmf, "     <DataItem Dimensions=\"%u %u\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", ny + 1, nx + 1);
    fprintf (xmf, "      %s:/common/x\n", hdf_filename.c_str ());
    fprintf (xmf, "     </DataItem>\n");
    fprintf (xmf, "     <DataItem Dimensions=\"%u %u\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", ny + 1, nx + 1);
    fprintf (xmf, "      %s:/common/y\n", hdf_filename.c_str ());
    fprintf (xmf, "     </DataItem>\n");
    fprintf (xmf, "   </Geometry>\n");
    fprintf (xmf, "   <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n");
  }

  void write_xdmf_xml_body (unsigned int nx, unsigned int ny, bool use_double_precision, const std::vector<std::string> &fields)
  {
    std::string hdf_filename = filename + ".h5";
    std::string xmf_filename = filename + ".xmf";

    fprintf (xmf, "     <Grid Name=\"T%lu\" GridType=\"Uniform\">\n", step);
    fprintf (xmf, "       <Time Value=\"%lu\"/>\n", step);
    fprintf (xmf, "       <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>\n");
    fprintf (xmf, "       <Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>\n");

    for (auto &field: fields)
    {
      fprintf (xmf, "       <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", field.c_str ());
      fprintf (xmf, "         <DataItem Dimensions=\"%u %u\" NumberType=\"Float\" Precision=\"%lu\" Format=\"HDF\">\n", ny, nx, use_double_precision ? sizeof (double) : sizeof (float));
      fprintf (xmf, "          %s:/simulation/%lu/%s\n", hdf_filename.c_str (), step, field.c_str ());
      fprintf (xmf, "         </DataItem>\n");
      fprintf (xmf, "       </Attribute>\n");
    }
    fprintf (xmf, "     </Grid>\n");
  }

  void write_xdmf_xml_tail ()
  {
    fprintf (xmf, "   </Grid>\n");
    fprintf (xmf, " </Domain>\n");
    fprintf (xmf, "</Xdmf>\n");
    fclose  (xmf);
  }
#endif

private:
  bool is_valid = false;

  std::size_t step {};

#if HDF5_BUILD
  hid_t file_id {};
  hid_t common_group_id {};
  hid_t simulation_group_id {};

  FILE *xmf = nullptr;
#endif

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
