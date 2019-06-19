//
// Created by egi on 6/15/19.
//

#include "core/config/configuration.h"

// TODO Use momento pattern to easily undo configuration changes

configuration::configuration () = default;

configuration_node& configuration::get_root ()
{
  return root;
}

const configuration_node& configuration::get_root () const
{
  return root;
}

unsigned int configuration::get_version () const
{
  return root.get_version ();
}
