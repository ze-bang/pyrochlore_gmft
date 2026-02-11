import os
import numpy as np
# os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.phase_diagram import *
import sys

SLURM_ID, SLURM_SIZE, MPI_SIZE = sys.argv[1:]

JPM_PARAM_SIZE = int(SLURM_SIZE) / 4
JOB_ID = int(SLURM_ID) - 1
flux_ind_ns_ind = JOB_ID // JPM_PARAM_SIZE
flux_ind = flux_ind_ns_ind//2
nS = flux_ind_ns_ind % 2

if flux_ind == 0:
    flux = np.zeros(4)
    flux_string = "0_flux"
else:
    flux = np.ones(4)*np.pi
    flux_string = "pi_flux"

Jpm_section = int(np.sqrt(JPM_PARAM_SIZE))
JPM_SECTION_ID = JOB_ID % JPM_PARAM_SIZE

Jpm_length = int(JPM_SECTION_ID) // Jpm_section
Jpm_width = int(JPM_SECTION_ID) % Jpm_section

Jpm_unit = 2/Jpm_section

Jpm_length_start = -1 + Jpm_length * Jpm_unit
Jpm_length_end = -1 + (Jpm_length+1) * Jpm_unit
Jpm_width_start = -1 + Jpm_width * Jpm_unit
Jpm_width_end = -1 + (Jpm_width+1) * Jpm_unit

# print(Jpm_length_start, Jpm_length_end, Jpm_width, Jpm_width, flux, nS, SLURM_ID, SLURM_SIZE, MPI_SIZE)
filename_here = "pyrochlore_XYZ_0_field_full_con/pyrochlore_xyz_" + flux_string + "_" + str(nS) + "_" + str(SLURM_ID) + "_out_of_" + str(SLURM_SIZE)
print(filename_here)
findXYZPhase_separate(float(Jpm_length_start), float(Jpm_length_end), float(Jpm_width_start), float(Jpm_width_end), int(np.sqrt(int(MPI_SIZE))), 30, 2, flux, filename_here, int(nS), symmetrized=False)
