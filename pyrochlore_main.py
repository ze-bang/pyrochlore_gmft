import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
import math
import time
import sys
from phase_diagram import *
from numba import jit
from misc_helper import *




#region graph dispersion

#endregion




graphdispersion(-1/3, 1, h001, 1, 2, 20, 20)
# graphdispersion(0.046,0, h111, 1, 2, 20, 20)
# graphdispersion(0.046, 0, h111, 1, 2, 20, 20)
# graphdispersion(0.05, 0, h111, 1, 2, 20, 20)
# graphdispersion(0.1, 0, h111, 1, 2, 20, 20)
# plt.show()

# findPhase(60,20, 20, "Files/phase_diagram.txt")

# PhaseMagtestH(0.0001, 0.25, 25, 0, 3, 25, h110, 35, 1, "0.txt")

# PhaseMagtestJP(-1, 0.1, 25, 0, 3, 25, h111, 35, 2, "0.txt")
#
#


#
# # SSSF(25, 0, np.array([1,1,1]),0.02,25, "SSSF_zero_0.02_h111=0")
#
# # SSSF(25, 0, h111,0.06,25, "SSSF_zero_0.06")

# SSSF(25, 0, np.array([1,1,1]),0.04,10, "SSSF_zero_test")
#
#

# samplegraph(100, ["SSSF_pi_0.02_DETAILED", "SSSF_pi_0.03_DETAILED", "SSSF_pi_0.04_DETAILED"])

# SSSF(100, 1, h001, hb110, -0.05, 15, "SSSF_pi_-0.05_DETAILED")
# SSSF(100, 0, h111, hb110, -0.20, 30, "SSSF_pi_-0.20_DETAILED")
# SSSF(100, 0, h111, hb110, -0.40, 30, "SSSF_pi_-0.40_DETAILED")



# SSSF(100, 0, h111, hb110, 0.02, 30, "SSSF_pi_0.02_DETAILED")
# SSSF(100, 0, h111, hb110, 0.03, 30, "SSSF_pi_0.03_DETAILED")
# SSSF(100, 0, h111, hb110, 0.04, 30, "SSSF_pi_0.04_DETAILED")



# SSSF(50, 0, np.array([1,1,1]),-0.40,30, "SSSF_pi_-0.40_DETAILED")

# SSSF(25, 0, h111,hb110, 0.04,10, "SSSF_zero_0.02")

# SSSF(100, 0.2, h111,hb110, -1/3,30, "SSSF_pi_-0.33_h111=0.2")
# SSSF(100, 0.3, h111,hb110, -1/3,30, "SSSF_pi_-0.33_h111=0.3")
# SSSF(100, 0.4, h111,hb110, -1/3,30, "SSSF_pi_-0.33_h111=0.4")
#
# SSSF(100, 0.2, h001,hb110, -1/3,30, "SSSF_pi_-0.33_h001=0.2")
# SSSF(100, 0.3, h001,hb110, -1/3,30, "SSSF_pi_-0.33_h001=0.3")
# SSSF(100, 0.4, h001,hb110, -1/3,30, "SSSF_pi_-0.33_h001=0.4")
#
# SSSF(100, 0.8, h110,hb110, -1/3,30, "SSSF_pi_-0.33_h110=0.8")
# SSSF(100, 1.2, h110,hb110, -1/3,30, "SSSF_pi_-0.33_h110=1.2")
# SSSF(100, 1.6, h110,hb110, -1/3,30, "SSSF_pi_-0.33_h110=1.6")
#
# SSSF(100, 0.2, h111,hb110, 0.02,30, "SSSF_zero_0.02_h111=0.2")
# SSSF(100, 0.3, h111,hb110, 0.02,30, "SSSF_zero_0.02_h111=0.3")
# SSSF(100, 0.4, h111,hb110, 0.02,30, "SSSF_zero_0.02_h111=0.4")
#
# SSSF(100, 0.2, h001,hb110, 0.02,30, "SSSF_zero_0.02_h001=0.2")
# SSSF(100, 0.3, h001,hb110, 0.02,30, "SSSF_zero_0.02_h001=0.3")
# SSSF(100, 0.4, h001,hb110, 0.02,30, "SSSF_zero_0.02_h001=0.4")
# #
# SSSF(100, 0.8, h110,hb110, 0.02,30, "SSSF_zero_0.02_h110=0.8")
# SSSF(100, 1.2, h110,hb110, 0.02,30, "SSSF_zero_0.02_h110=1.2")
# SSSF(100, 1.6, h110,hb110, 0.02,30, "SSSF_zero_0.02_h110=1.6")

# DSSF(0.02, 0, h111,-1/3, "DSSF_-0.33_detailed", 30, 0.02)



#
# graphPhase("Files/phase_diagram.txt")


# findPhaseMag(-1, 0.1, 70, 0, 1, 40, h111, 25, 2, "phase_test_111_kappa=2")
# findPhaseMag(-1, 0.1, 70, 0, 1, 40, h001, 25, 2, "phase_test_001_kappa=2")
# findPhaseMag(-1, 0.1, 70, 0, 4, 40, h110, 25, 2, "phase_test_110_kappa=2")
#
# DSSF(0.01,0, h111, 0.046, "DSSF_0.046_detailed", 30, 0.02)


# TWOSPINCON(40, 0, h111, -1/3, 25, "TSC_-0.33_h111=0")
# TWOSPINCON(40, 0.2, h111, -1/3, 25, "TSC_-0.33_h111=0.2")
# TWOSPINCON(40, 1, h111, -1/3, 25, "TSC_-0.33_h111=1.0")
#
# TWOSPINCON(40, 0, h001, -1/3, 25, "TSC_-0.33_h001=0")
# TWOSPINCON(40, 0.2, h001, -1/3, 25, "TSC_-0.33_h001=0.2")
# TWOSPINCON(40, 1, h001, -1/3, 25, "TSC_-0.33_h001=1.0")
#
# TWOSPINCON(40, 0, h110, -1/3, 25, "TSC_-0.33_h110=0")
# TWOSPINCON(40, 0.2, h110, -1/3, 25, "TSC_-0.33_h110=0.2")
# TWOSPINCON(40, 1, h110, -1/3, 25, "TSC_-0.33_h110=1.0")
#
# graphPhase("Files/phase_diagram.txt")


findPhaseMag(-0.5, 0.1, 300, 0, 1, 100, h111, 25, 2, "phase_test_111_kappa=2")
findPhaseMag(-0.5, 0.1, 300, 0, 1, 100, h001, 25, 2, "phase_test_001_kappa=2")
findPhaseMag(-0.5, 0.1, 300, 0, 4, 100, h110, 25, 2, "phase_test_110_kappa=2")


# graphMagPhase("phase_test_111_kappa=1", 0.25,3)
# graphMagPhase("phase_test_001_kappa=1", 0.25,3)
# graphMagPhase("phase_test_110_kappa=1", 0.25,12)

# graphMagPhase("phase_test_001", 0.25,3)
# graphMagPhase("phase_test_110", 0.25,12)