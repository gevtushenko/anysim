//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_CONFIGURATION_READER_H
#define ANYSIM_CONFIGURATION_READER_H

#include <string>
#include <memory>

class json_wrapper;
class project_manager;

class configuration_reader
{
public:
  configuration_reader () = delete;
  ~configuration_reader ();

  explicit configuration_reader (const std::string &filename);

  bool initialize_project (project_manager &pm);
  bool is_valid () { return data != nullptr; }

private:
  /// Required options
  std::string solver_name;
  std::string project_name;
  std::string initializer_script;
  double max_time = 1.0;
  bool use_double_precision = true;

  /// Solver options
  std::unique_ptr<json_wrapper> data;
};

#endif //ANYSIM_CONFIGURATION_READER_H
