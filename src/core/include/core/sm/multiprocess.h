//
// Created by egi on 7/6/19.
//

#ifndef ANYSIM_MULTIPROCESS_H
#define ANYSIM_MULTIPROCESS_H

#include "cpp/common_funcs.h"

#include <memory>

constexpr inline unsigned int get_main_process () { return 0; }
inline bool is_main_process (int rank) { return rank == get_main_process (); }

class transaction_id;
class transaction_id_holder : private non_copyable<transaction_id>
{
public:
  transaction_id_holder ();

private:
  std::unique_ptr<transaction_id> request;
};

class multiprocess
{
public:
  multiprocess (int argc, char *argv[]);
  ~multiprocess ();

  int get_rank () const { return rank; }
  int get_size () const { return size; }

private:
  int rank = 0;
  int size = 1;
};

#endif //ANYSIM_MULTIPROCESS_H
