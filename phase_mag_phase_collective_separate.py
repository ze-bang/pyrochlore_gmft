import os
import numpy as np
# os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from observables import *
import sys

# hmin, hmax, field_dir, Jpmpm, SLURM_ID, SLURM_SIZE, mpi_size = sys.argv[1:]

hmin, hmax, field_dir, Jpmpm, SLURM_ID, SLURM_SIZE, mpi_size = 0, 0.5, "111", 0.3, 2, 144, 441


hmin = float(hmin)
hmax = float(hmax)
Jpmpm = float(Jpmpm)

if field_dir == "110":
    h_dir = h110
    JPMH_PARAM_SIZE = int(SLURM_SIZE) / 4
elif field_dir == "111":
    h_dir = h111
    if Jpmpm == 0:
        JPMH_PARAM_SIZE = int(SLURM_SIZE) / 3
    else:
        JPMH_PARAM_SIZE = int(SLURM_SIZE) / 2
else:
    h_dir = h001
    JPMH_PARAM_SIZE = int(SLURM_SIZE) / 2

JOB_ID = int(SLURM_ID) - 1
flux_ind_ns_ind = JOB_ID // JPMH_PARAM_SIZE
JPMH_SECTION_ID = JOB_ID % JPMH_PARAM_SIZE
FF = False

if Jpmpm == 0 and field_dir == "111":
    if flux_ind_ns_ind == 0:
        flux = np.zeros(4)
    elif flux_ind_ns_ind == 1:
        flux = np.ones(4)*np.pi
    elif flux_ind_ns_ind == 2:
        flux = FFFluxGen(np.pi/6)
        FF = True
else:
    if flux_ind_ns_ind == 0:
        flux = np.zeros(4)
    elif flux_ind_ns_ind == 1:
        flux = np.ones(4)*np.pi
    elif flux_ind_ns_ind == 2:
        flux = zppz
    else:
        flux = pzzp


Jpm_min = -0.5 + abs(float(Jpmpm))
Jpm_section = int(np.sqrt(int(JPMH_PARAM_SIZE)))
Jpm_length = int(JPMH_SECTION_ID) // Jpm_section
h_length = int(JPMH_SECTION_ID) % Jpm_section

Jpm_unit = (0.1 - Jpm_min)/Jpm_section
h_unit = (hmax - hmin) / Jpm_section

Jpm_length_start = Jpm_min + Jpm_length * Jpm_unit
Jpm_length_end = Jpm_min + (Jpm_length+1) * Jpm_unit
h_length_start = hmin + h_length * h_unit
h_length_end = hmin + (h_length+1) * h_unit

if Jpmpm == 0:
    filename = "pyrochlore_mag_phase_" + str(field_dir) + "_Jpmpm=0/Jpm_" + str(Jpm_length_start) + "_" + str(Jpm_length_end) + "_h_" + str(h_length_start) + "_" + str(h_length_end) + "_" + str(SLURM_ID) + "_out_of_" + str(SLURM_SIZE)
else:
    filename = "pyrochlore_mag_phase_" + str(field_dir) + "/Jpm_" + str(Jpm_length_start) + "_" + str(Jpm_length_end) + "_h_" + str(h_length_start) + "_" + str(h_length_end) + "_" + str(SLURM_ID) + "_out_of_" + str(SLURM_SIZE)


nK = int(np.sqrt(int(mpi_size)))

findPhaseMag_separate(Jpm_length_start, Jpm_length_end, nK, h_length_start, h_length_end, nK, h_dir, flux, 30, 2, filename, Jpmpm=Jpmpm, FF=FF)
