//
// Created by egi on 6/28/19.
//

#ifndef ANYSIM_HDF5_WRITER_H
#define ANYSIM_HDF5_WRITER_H

#include <memory>
#include <string>

#include "core/sm/result_extractor.h"
#include "core/pm/project_manager.h"

class hdf5_writer : public result_extractor
{
  class hdf5_impl;

public:
  hdf5_writer (const std::string &file, project_manager &pm_arg);
  ~hdf5_writer ();

  void extract (
        unsigned int thread_id,
        unsigned int threads_count,
        thread_pool &threads) final;
  bool open ();
  bool close ();

private:
  std::unique_ptr<hdf5_impl> implementation;
};

#endif //ANYSIM_HDF5_WRITER_H
