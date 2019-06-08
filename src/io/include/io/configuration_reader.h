//
// Created by egi on 6/8/19.
//

#ifndef ANYSIM_CONFIGURATION_READER_H
#define ANYSIM_CONFIGURATION_READER_H

#include <string>

class confituration_reader
{
public:
  confituration_reader () = delete;

  explicit confituration_reader (const std::string &filename);
};

#endif //ANYSIM_CONFIGURATION_READER_H
