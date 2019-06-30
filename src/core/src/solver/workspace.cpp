//
// Created by egi on 6/16/19.
//

#include "core/solver/workspace.h"

#include <cstddef>

#ifdef GPU_BUILD
#include <cuda_runtime.h>
#endif

class memory_object
{
public:
  virtual ~memory_object () = default;
  virtual bool allocate (std::size_t bytes) = 0;
  virtual bool destroy () = 0;

  void *get () { return ptr; }
  const void *get () const { return ptr; }

  std::size_t get_size () const { return size; }
  memory_holder_type get_holder () const { return holder; }

protected:
  memory_holder_type holder = memory_holder_type::host;

  void *ptr = nullptr;
  std::size_t size = 0;
};

class cpu_memory_object : public memory_object
{
public:
  bool allocate (std::size_t bytes) override
  {
    ptr = malloc (bytes);
    size = bytes;
    return false;
  }

  bool destroy () override
  {
    if (ptr)
      free (ptr);

    return false;
  }

  ~cpu_memory_object () override { destroy (); }
};

#ifdef GPU_BUILD
class gpu_memory_object : public memory_object
{
public:
  gpu_memory_object () : memory_object () { holder = memory_holder_type::device; }

  bool allocate (std::size_t bytes) override
  {
    cudaMalloc (&ptr, bytes);
    size = bytes;
    return false;
  }

  bool destroy () override
  {
    if (ptr)
      cudaFree (ptr);

    return false;
  }

  ~gpu_memory_object () override { destroy (); }
};
#endif

class pinned_memory
{
public:
  ~pinned_memory ()
  {
#ifdef GPU_BUILD
    if (ptr)
    {
      cudaFreeHost (ptr);

      size = 0;
      ptr = nullptr;
    }
#endif
  }

  void allocate (std::size_t size_arg)
  {
#ifdef GPU_BUILD
    if (size_arg > size)
    {
      size = size_arg;
      cudaMallocHost (&ptr, size);
    }
#else
    (void) size_arg;
#endif
  }

  std::byte *get () { return ptr; }
private:
  std::size_t size = 0;
  std::byte *ptr = nullptr;
};


class layered_memory_object
{
public:
  layered_memory_object () = delete;
  layered_memory_object (unsigned int layers_count_arg, memory_holder_type holder)
    : layers_count (layers_count_arg)
    , storage (new std::unique_ptr<memory_object>[layers_count])
  {
    if (holder == memory_holder_type::host)
    {
      for (unsigned int lid = 0; lid < layers_count; lid++)
      {
        storage[lid] = std::make_unique<cpu_memory_object> ();
      }
    }
#ifdef GPU_BUILD
    else
    {
      for (unsigned int lid = 0; lid < layers_count; lid++)
        {
          storage[lid] = std::make_unique<gpu_memory_object> ();
        }
    }
#endif
  }

  bool allocate (std::size_t bytes)
  {
    for (unsigned int lid = 0; lid < layers_count; lid++)
      if (storage[lid]->allocate (bytes))
        return true;

    return false;
  }

  void set_active_layer (unsigned int lid)
  {
    active_layer = lid;
  }

  void *get_layer (unsigned int lid) { return storage[lid]->get (); }
  const void *get_layer (unsigned int lid) const { return storage[lid]->get (); }

  void *get_active_layer () { return get_layer (active_layer); }
  const void *get_active_layer () const { return get_layer (active_layer); }

  memory_object *get_memory_object () { return storage[active_layer].get (); }

private:
  unsigned int active_layer = 0;
  const unsigned int layers_count = 0;
  std::unique_ptr<std::unique_ptr<memory_object>[]> storage;
};

workspace::workspace ()
  : temporal_buffer (new pinned_memory ())
{ }

workspace::~workspace () = default;

bool workspace::allocate (
    const std::string &name,
    memory_holder_type holder,
    std::size_t bytes,
    unsigned int layouts_count)
{
  storage[name] = std::make_unique<layered_memory_object> (layouts_count, holder);
  return storage[name]->allocate (bytes);
}

const void* workspace::get (const std::string &name, unsigned int lid) const
{
  auto it = storage.find (name);

  if (it != storage.end ())
    return it->second->get_layer (lid);
  return nullptr;
}

void* workspace::get (const std::string &name, unsigned int lid)
{
  auto it = storage.find (name);

  if (it != storage.end ())
    return it->second->get_layer (lid);
  return nullptr;
}

void* workspace::get (const std::string &name)
{
  auto it = storage.find (name);

  if (it != storage.end ())
    return it->second->get_active_layer ();
  return nullptr;
}

const void* workspace::get (const std::string &name) const
{
  auto it = storage.find (name);

  if (it != storage.end ())
    return it->second->get_active_layer ();
  return nullptr;
}

void workspace::set_active_layer (const std::string &name, unsigned int lid)
{
  auto it = storage.find (name);

  if (it != storage.end ())
    it->second->set_active_layer (lid);
}

const void *workspace::get_host_copy (const std::string &name) const
{
  auto it = storage.find (name);

  if (it != storage.end ())
  {
    auto mo = it->second->get_memory_object ();

#ifdef GPU_BUILD
    const auto size = mo->get_size ();

    if (mo->get_holder () == memory_holder_type::device)
    {
      temporal_buffer->allocate (size);
      cudaMemcpy (temporal_buffer->get (), mo->get (), size, cudaMemcpyDeviceToHost);
      return temporal_buffer->get ();
    }
#endif

    return mo->get ();
  }

  return nullptr;
}
