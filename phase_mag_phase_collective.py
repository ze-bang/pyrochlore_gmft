import os
import numpy as np
# os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from observables import *
import sys

hmin, hmax, field_dir, Jpmpm, SLURM_ID, SLURM_SIZE, mpi_size = sys.argv[1:]

SLURM_ID = int(SLURM_ID) - 1

hmin = float(hmin)
hmax = float(hmax)
Jpmpm = float(Jpmpm)

Jpm_min = -0.5 + abs(float(Jpmpm))
Jpm_section = int(np.sqrt(int(SLURM_SIZE)))
Jpm_length = int(SLURM_ID) // Jpm_section
h_length = int(SLURM_ID) % Jpm_section

Jpm_unit = (0.1 - Jpm_min)/Jpm_section
h_unit = (hmax - hmin) / Jpm_section

Jpm_length_start = Jpm_min + Jpm_length * Jpm_unit
Jpm_length_end = Jpm_min + (Jpm_length+1) * Jpm_unit
h_length_start = hmin + h_length * h_unit
h_length_end = hmin + (h_length+1) * h_unit

filename = "pyrochlore_mag_phase_" + str(field_dir) + "/Jpm_" + str(Jpm_length_start) + "_" + str(Jpm_length_end) + "_h_" + str(h_length_start) + "_" + str(h_length_end) + "_" + str(SLURM_ID) + "_out_of_" + str(SLURM_SIZE)

if field_dir == "110":
    h_dir = h110
elif field_dir == "111":
    h_dir = h111
else:
    h_dir = h001

nK = int(np.sqrt(int(mpi_size)))

if field_dir == "110":
    findPhaseMag110(Jpm_length_start, Jpm_length_end, nK, h_length_start, h_length_end, nK, h_dir, 30, 2, filename, Jpmpm=Jpmpm)
else:
    findPhaseMag111(Jpm_length_start, Jpm_length_end, nK, h_length_start, h_length_end, nK, h_dir, 30, 2, filename, Jpmpm=Jpmpm)
