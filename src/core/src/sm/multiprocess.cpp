//
// Created by egi on 7/6/19.
//

#include "core/sm/multiprocess.h"
#include "cpp/common_funcs.h"

#ifdef MPI_BUILD
#include <mpi.h>
#endif

#include <iostream>

multiprocess::multiprocess (int argc, char **argv)
{
  cpp_unreferenced (argc, argv);
#ifdef MPI_BUILD
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  if (is_main_process (rank))
    std::cout << "Run " << size << " processes" << std::endl;
#endif
}

multiprocess::~multiprocess ()
{
#ifdef MPI_BUILD
  MPI_Finalize ();
#endif
}
