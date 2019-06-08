//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_CONFIGURATION_READER_H
#define ANYSIM_CONFIGURATION_READER_H

#include <string>
#include <memory>

class json_wrapper;
class project_manager;

class confituration_reader
{
public:
  confituration_reader () = delete;
  ~confituration_reader ();

  explicit confituration_reader (const std::string &filename);

  void initialize_project (project_manager &pm);
  bool is_valid () { return data != nullptr; }

private:
  double max_time = 3.14E-8;
  double geometry_width = 5.0;
  double geometry_height = 5.0;
  unsigned int cells_per_lambda = 40;

  std::unique_ptr<json_wrapper> data;
};

#endif //ANYSIM_CONFIGURATION_READER_H
