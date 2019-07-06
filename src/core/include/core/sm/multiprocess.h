//
// Created by egi on 7/6/19.
//

#ifndef ANYSIM_MULTIPROCESS_H
#define ANYSIM_MULTIPROCESS_H

constexpr inline unsigned int get_main_process () { return 0; }
inline bool is_main_process (int rank) { return rank == get_main_process (); }

class multiprocess
{
public:
  multiprocess (int argc, char *argv[]);
  ~multiprocess ();

private:
  int rank = 0;
  int size = 1;
};

#endif //ANYSIM_MULTIPROCESS_H
