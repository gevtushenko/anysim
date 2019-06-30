//
// Created by egi on 6/16/19.
//

#ifndef ANYSIM_WORKSPACE_H
#define ANYSIM_WORKSPACE_H

#include <cstddef>
#include <memory>
#include <map>

enum class memory_holder_type
{
  host, device
};

class pinned_memory;
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

  /**
   * If memory for name is stored on GPU it'll be copied in temporal
   * buffer. In other case pointer to memory object will be returned.
   */
  const void *get_host_copy (const std::string &name) const;

private:
  std::unique_ptr<pinned_memory> temporal_buffer;
  std::map<std::string, std::unique_ptr<layered_memory_object>> storage;
};

#endif //ANYSIM_WORKSPACE_H
