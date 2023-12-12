import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *

# DSSF(0.1, 0.2, h001,-0.1, "DSSF_-0.1_h001=0.2_crude", 35, 0.2)
# DSSF(0.1, 0.2, h1b10,-0.1, "DSSF_-0.1_h1b10=0.2_crude", 35, 0.2)



findXYZPhase(0, 1, 0, 1, 50, 35, 2, "XYZ_0_field_upper_quadrant")
#findPhaseMag(-0.5, 0.1, 100, 0, 2, 100, h001, 26, 2, "phase_001_kappa=2_crude_faster")

# findPhaseMag_pi(-0.5, 0.1, 300, 0, 4, 150, h1b10, 25, 2, "phase_1b10_kappa=2_pi_flux")
# findPhaseMag_zero(-0.5, 0.1, 300, 0, 4, 150, h1b10, 25, 2, "phase_1b10_kappa=2_zero_flux")
# findPhaseMag_pi(-0.5, 0.1, 300, 0, 1, 150, h001, 25, 2, "phase_1b10_kappa=2_pi_flux")
# findPhaseMag_zero(-0.5, 0.1, 300, 0, 1, 150, h001, 25, 2, "phase_1b10_kappa=2_zero_flux")

# SSSF(100, 0.8, h1b10,hb110, -0.1, 35, "SSSF_pi_-0.1_h1b10=0.8")
# SSSF(100, 1.0, h1b10,hb110, -0.1, 35, "SSSF_pi_-0.1_h1b10=1.0")
# SSSF(100, 1.2, h1b10,hb110, -0.1, 35, "SSSF_pi_-0.1_h1b10=1.2")

# SSSF(100, 0.8, h110,hb110, -0.1, 35, "SSSF_pi_-0.1_h110=0.8")
# SSSF(100, 1.0, h110,hb110, -0.1, 35, "SSSF_pi_-0.1_h110=1.0")
# SSSF(100, 1.2, h110,hb110, -0.1, 35, "SSSF_pi_-0.1_h110=1.2")