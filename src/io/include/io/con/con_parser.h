//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_CON_PARSER_H
#define ANYSIM_CON_PARSER_H

#include <memory>

class argagg_wrapper;
class project_manager;

class con_parser
{
public:
  con_parser ();
  ~con_parser ();

  /**
   * @return true if AnySim has to exit after parse call
   */
  bool parse (int argc, char *argv[], bool require_configuration, project_manager &pm);

private:
  std::unique_ptr<argagg_wrapper> parser_wrapper;
};

#endif //ANYSIM_CON_PARSER_H
