//
// Created by egi on 5/9/19.
//

#ifndef FDTD_VTK_H
#define FDTD_VTK_H

#include <iostream>
#include <stdio.h>
#include <string>
#include <chrono>

template <class float_type>
void write_vtk (
  const std::string &filename,
  const float_type dx,
  const float_type dy,
  const unsigned int nx,
  const unsigned int ny,
  const float_type *e)
{
  std::cout << "Write step ";
  const auto begin = std::chrono::high_resolution_clock::now ();
  FILE * f = fopen (filename.c_str (), "w");

  fprintf (f, "# vtk DataFile Version 3.0\n");
  fprintf (f, "vtk output\n");
  fprintf (f, "ASCII\n");
  fprintf (f, "DATASET UNSTRUCTURED_GRID\n");
  fprintf (f, "POINTS %u double\n", nx * ny * 4);

  for (unsigned int j = 0; j < ny; j++)
    {
      for (unsigned int i = 0; i < nx; i++)
        {
          fprintf (f, "%lf %lf 0.0\n", dx * (i + 0), dy * (j + 0) );
          fprintf (f, "%lf %lf 0.0\n", dx * (i + 1), dy * (j + 0) );
          fprintf (f, "%lf %lf 0.0\n", dx * (i + 1), dy * (j + 1) );
          fprintf (f, "%lf %lf 0.0\n", dx * (i + 0), dy * (j + 1) );
        }
    }

  fprintf (f, "CELLS %u %u\n", nx * ny, nx * ny * 5);

  for (unsigned int j = 0; j < ny; j++)
    {
      for (unsigned int i = 0; i < nx; i++)
        {
          const unsigned int point_offset = (j * nx + i) * 4;
          fprintf (f, "4 %u %u %u %u\n", point_offset + 0, point_offset + 1, point_offset + 2, point_offset + 3);
        }
    }

  fprintf (f, "CELL_TYPES %u\n", nx * ny);
  for (unsigned int i = 0; i < nx * ny; i++)
    fprintf (f, "9\n");

  fprintf (f, "CELL_DATA %u\n", nx * ny);
  fprintf (f, "SCALARS Ez double 1\n");
  fprintf (f, "LOOKUP_TABLE default\n");

  for (unsigned int i = 0; i < nx * ny; i++)
    fprintf (f, "%lf\n", e[i]);

  fclose (f);

  const auto end = std::chrono::high_resolution_clock::now ();
  const std::chrono::duration<double> duration = end - begin;
  std::cout << " in " << duration.count () << "s\n";
}


#endif //FDTD_VTK_H
