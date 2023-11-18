import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *


# DSSF(0.01, 0.2, h111,-0.1, "DSSF_-0.1_h111=0.2_detailed", 35, 0.02)
# DSSF(0.01, 0.2, h001,-0.1, "DSSF_-0.1_h001=0.2_detailed", 35, 0.02)
# DSSF(0.01, 0.2, h1b10,-0.1, "DSSF_-0.1_h1b10=0.2_detailed", 35, 0.02)


findPhaseMag(-0.1, 0.1, 100, 0, 0.3, 100, h110, 26, 2, "phase_110_kappa=2_zoomed_in")
findPhaseMag(-0.1, 0.1, 100, 0, 0.3, 100, h001, 26, 2, "phase_001_kappa=2_zoomed_in")
findPhaseMag(-0.1, 0.1, 100, 0, 0.3, 100, h111, 26, 2, "phase_111_kappa=2_zoomed_in")
findPhaseMag(-0.1, 0.1, 100, 0, 0.3, 100, h1b10, 26, 2, "phase_1b10_kappa=2_zoomed_in")

# SSSF(50, 0.2, h111,hb110, -0.1,31, "SSSF_pi_-0.1_h111=0.2")
# SSSF(50, 0.3, h111,hb110, -0.1,31, "SSSF_pi_-0.1_h111=0.3")
# SSSF(50, 0.4, h111,hb110, -0.1,31, "SSSF_pi_-0.1_h111=0.4")
# #
# SSSF(50, 0.2, h001,hb110, -0.1,31, "SSSF_pi_-0.1_h001=0.2")
# SSSF(50, 0.3, h001,hb110, -0.1,31, "SSSF_pi_-0.1_h001=0.3")
# SSSF(50, 0.4, h001,hb110, -0.1,31, "SSSF_pi_-0.1_h001=0.4")

# SSSF(50, 0.4, h110,hb110, -0.1,31, "SSSF_pi_-0.1_h110=0.4")
# SSSF(50, 0.6, h110,hb110, -0.1,31, "SSSF_pi_-0.1_h110=0.6")
# SSSF(50, 0.8, h110,hb110, -0.1,31, "SSSF_pi_-0.1_h110=0.8")
#
# SSSF(100, 0.2, h111,hb110, 0.02,50, "SSSF_zero_0.02_h111=0.2")
# SSSF(100, 0.3, h111,hb110, 0.02,50, "SSSF_zero_0.02_h111=0.3")
# SSSF(100, 0.4, h111,hb110, 0.02,50, "SSSF_zero_0.02_h111=0.4")

# SSSF(100, 0.2, h001,hb110, 0.02,50, "SSSF_zero_0.02_h001=0.2")
# SSSF(100, 0.3, h001,hb110, 0.02,50, "SSSF_zero_0.02_h001=0.3")
# SSSF(100, 0.4, h001,hb110, 0.02,50, "SSSF_zero_0.02_h001=0.4")
# #
# SSSF(100, 0.4, h110,hb110, 0.02,50, "SSSF_zero_0.02_h110=0.4")
# SSSF(100, 0.6, h110,hb110, 0.02,50, "SSSF_zero_0.02_h110=0.6")
# SSSF(100, 0.8, h110,hb110, 0.02,50, "SSSF_zero_0.02_h110=0.8")
