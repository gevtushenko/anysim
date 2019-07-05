from anysim_py import *

x_0 = 1.0
y_0 = 1.5

for cell_id in range (topology.get_cells_count ()):
    if geometry.get_cell_center_x (cell_id) < x_0:
        fields["rho"][cell_id] = 1.0
        fields["p"][cell_id] = 1.0
        fields["u"][cell_id] = 0.0
        fields["v"][cell_id] = 0.0
    else:
        if geometry.get_cell_center_y (cell_id) < y_0:
            fields["rho"][cell_id] = 1.0
            fields["p"][cell_id] = 0.1
            fields["u"][cell_id] = 0.0
            fields["v"][cell_id] = 0.0
        else:
            fields["rho"][cell_id] = 0.125
            fields["p"][cell_id] = 0.1
            fields["u"][cell_id] = 0.0
            fields["v"][cell_id] = 0.0
