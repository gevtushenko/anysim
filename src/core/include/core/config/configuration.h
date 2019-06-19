//
// Created by egi on 6/15/19.
//

#ifndef ANYSIM_CONFIGURATION_H
#define ANYSIM_CONFIGURATION_H

#include "core/config/configuration_node.h"

/**
 * Configuration tree. It's supposed that accessing non constant
 * version of get_root updates configuration.
 *
 */
class configuration
{
public:
  configuration ();

  configuration_node &get_root ();
  const configuration_node &get_root () const;

  unsigned int get_version () const;

private:
  configuration_node root;
};

#endif //ANYSIM_CONFIGURATION_H
