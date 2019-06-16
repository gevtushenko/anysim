//
// Created by egi on 6/16/19.
//

#ifndef ANYSIM_WORKSPACE_H
#define ANYSIM_WORKSPACE_H

#include <memory>
#include <map>

enum class memory_holder_type
{
  host, device
};

class layered_memory_object;

class workspace
{
public:
  workspace ();
  ~workspace ();

  bool allocate (
      const std::string &name,
      memory_holder_type holder,
      std::size_t bytes,
      unsigned int layouts_count = 1);

  void *get (const std::string &name, unsigned int lid);
  const void *get (const std::string &name, unsigned int lid) const;

  void *get (const std::string &name);
  const void *get (const std::string &name) const;

  void set_active_layer (const std::string &name, unsigned int lid);

private:
  std::map<std::string, std::unique_ptr<layered_memory_object>> storage;
};

#endif //ANYSIM_WORKSPACE_H
