import os
import numpy as np
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
import sys

SLURM_ID, SLURM_SECTIONS = sys.argv[1:]

JPM_PARAM_SIZE = int(SLURM_SECTIONS) / 4
flux_ind_ns_ind = int(SLURM_ID) // JPM_PARAM_SIZE
flux_ind = flux_ind_ns_ind//2
nS = flux_ind_ns_ind % 2

if flux_ind == 0:
    flux = np.zeros(4)
    flux_string = "0_flux"
else:
    flux = np.ones(4)*np.pi
    flux_string = "pi_flux"

Jpm_section = int(np.sqrt(JPM_PARAM_SIZE))
Jpm_length = int(SLURM_ID) // Jpm_section
Jpm_width = int(SLURM_ID) % Jpm_section

Jpm_unit = 2/Jpm_section

Jpm_length_start = -1 + Jpm_length * Jpm_unit
Jpm_length_end = -1 + (Jpm_length+1) * Jpm_unit
Jpm_width_start = -1 + Jpm_width * Jpm_unit
Jpm_width_end = -1 + (Jpm_width+1) * Jpm_unit



print(Jpm_length_start, Jpm_length_end, Jpm_width, Jpm_width, flux, nS)
filename_here = "pyrochlore_xyz_" + flux_string + "_" + nS + "_" + SLURM_ID + "_out_of_" + SLURM_SECTIONS
# findXYZPhase_separate(float(Jpm_length_start), float(Jpm_length_end), float(Jpm_width_start), float(Jpm_width_end), 20, 30, 2, flux, filename_here, int(nS), symmetrized=False)
