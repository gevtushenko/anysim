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
      hid_t type,
      unsigned int cells_count)
  {
    hsize_t dims[3];
    dims[0] = cells_count;

    hid_t dataspace_id = H5Screate_simple(1, dims, nullptr);
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
      const unsigned int cells_count = solver_grid.get_cells_number ();

      if (step == 0)
      {
        const std::string v_group_name = "/common/vertices";
        const std::string t_group_name = "/common/topology";

        std::unique_ptr<int[]> topology (new int[cells_count * 4]);
        for (unsigned int c = 0; c < cells_count; c++)
          {
            for (unsigned int i = 0; i < 4; i++)
              topology[4 * c + i] = 4 * c + i;
          }

        write_field (solver_grid.get_vertices_data (), v_group_name, H5T_NATIVE_FLOAT, cells_count * 4 * 2);
        write_field (topology.get (), t_group_name, H5T_NATIVE_INT, cells_count * 4);
        write_xdmf_xml_head (cells_count);
      }

      const bool use_double_precision = pm.is_double_precision_used ();
      hid_t type = use_double_precision ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
      write_xdmf_xml_body (cells_count, use_double_precision, solver_grid.get_fields_names());

      const std::string time_step_group_name = "/simulation/" + std::to_string (step++);
      hid_t time_step_group_id = H5Gcreate2 (file_id, time_step_group_name.c_str (), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      for (auto &field: solver_grid.get_fields_names())
        write_field (solver_workspace.get_host_copy (field), time_step_group_name + "/" + field, type, cells_count);

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

  void write_xdmf_xml_head (unsigned int cells_number)
  {
    std::string hdf_filename = filename + ".h5";
    std::string xmf_filename = filename + ".xmf";
    xmf = fopen(xmf_filename.c_str (), "w");
    fprintf (xmf, "<?xml version=\"1.0\" ?>\n");
    fprintf (xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf (xmf, "<Xdmf Version=\"2.0\">\n");
    fprintf (xmf, " <Domain>\n");
    fprintf (xmf, "   <Topology TopologyType=\"Quadrilateral\" NumberOfElements=\"%u\">\n", cells_number);
    fprintf (xmf, "     <DataItem Dimensions=\"%u 4\" NumberType=\"Int\" Format=\"HDF\">\n", cells_number);
    fprintf (xmf, "      %s:/common/topology\n", hdf_filename.c_str ());
    fprintf (xmf, "     </DataItem>\n");
    fprintf (xmf, "   </Topology>\n");
    fprintf (xmf, "   <Geometry GeometryType=\"XY\">\n");
    fprintf (xmf, "     <DataItem Dimensions=\"%u 2\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", cells_number * 4);
    fprintf (xmf, "      %s:/common/vertices\n", hdf_filename.c_str ());
    fprintf (xmf, "     </DataItem>\n");
    fprintf (xmf, "   </Geometry>\n");
    fprintf (xmf, "   <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n");
  }

  void write_xdmf_xml_body (unsigned int cells_count, bool use_double_precision, const std::vector<std::string> &fields)
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
      fprintf (xmf, "         <DataItem Dimensions=\"%u\" NumberType=\"Float\" Precision=\"%lu\" Format=\"HDF\">\n", cells_count, use_double_precision ? sizeof (double) : sizeof (float));
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
