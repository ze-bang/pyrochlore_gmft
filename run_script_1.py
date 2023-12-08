import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import pyrochlore_dispersion_pi as pygang
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
import math
import time
import sys
from phase_diagram import *
from numba import jit
from misc_helper import *
import netCDF4 as nc


# findXYZPhase(-1, 1, 20, 26, 2, "XYZ_0_field_more")

flux = np.array([np.pi/4, -3*np.pi/4, np.pi/4, np.pi/4])


findPhaseMag(-0.05, 0.05, 100, 0, 0.3, 100, h110, 26, 2, flux, "FF_phase_110_kappa=2_zoomed_in_more")
findPhaseMag(-0.05, 0.05, 100, 0, 0.3, 100, h001, 26, 2, flux, "FF_phase_001_kappa=2_zoomed_in_more")
findPhaseMag(-0.05, 0.05, 100, 0, 0.3, 100, h111, 26, 2, flux, "FF_phase_111_kappa=2_zoomed_in_more")
findPhaseMag(-0.05, 0.05, 100, 0, 0.3, 100, h1b10, 26, 2, flux, "FF_phase_1b10_kappa=2_zoomed_in_more")


# findPhaseMag_pi_zero(-0.5, 0, 300, 0, 1, 150, h111, 25, 2, "phase_111_kappa=2_complete")
# findPhaseMag_pi_zero(-0.5, 0, 300, 0, 1, 150, h001, 25, 2, "phase_001_kappa=2_complete")
# findPhaseMag_pi_zero(-0.5, 0, 300, 0, 4, 150, h110, 25, 2, "phase_110_kappa=2_complete")

# SSSF(100, 0.8, h001,hb110, -0.1, 35, "SSSF_pi_-0.1_h001=0.8")
# SSSF(100, 0.3, h001,hb110, -0.1, 35, "SSSF_pi_-0.1_h001=0.3")
# SSSF(100, 0.4, h001,hb110, -0.1, 35, "SSSF_pi_-0.1_h001=0.4")