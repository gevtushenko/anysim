from anysim_py import *

for cell_id in range (topology.get_cells_count ()):
    if geometry.get_cell_center_x (cell_id) < 5:
        if geometry.get_cell_center_y (cell_id) > 1:
            print(cell_id)
            fields["rho"][cell_id] = 1.4
