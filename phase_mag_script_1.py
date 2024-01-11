import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
# import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *




flux = np.array([np.pi/4, -3*np.pi/4, np.pi/4, np.pi/4])

# findPhaseMag(-0.5, 0.1, 100, 0, 4, 100, h110, 26, 2, flux, "FF_phase_110_kappa=2")
# findPhaseMag(-0.5, 0.1, 100, 0, 1, 100, h001, 26, 2, flux, "FF_phase_001_kappa=2")
findPhaseMag(-0.5, 0.1, 100, 0, 1, 100, h111, 30, 2, flux, "FF_phase_111_kappa=2")
# findPhaseMag(-0.5, 0.1, 100, 0, 4, 100, h1b10, 26, 2, flux, "FF_phase_1b10_kappa=2")



# SSSF(100, 0, h111, hb110, 0.02, 35, "SSSF_zero_0.02_DETAILED")
# SSSF(100, 0, h111, hb110, 0.03, 35, "SSSF_zero_0.03_DETAILED")
# SSSF(100, 0, h111, hb110, 0.04, 35, "SSSF_zero_0.04_DETAILED")



# SSSF(50, 0, np.array([1,1,1]),-0.40,30, "SSSF_pi_-0.40_DETAILED")

# SSSF(25, 0, h111,hb110, 0.04,10, "SSSF_zero_0.02")

# SSSF(100, 0.4, h111,hb110, -0.1, 35, "SSSF_pi_-0.1_h111=0.4")
# SSSF(100, 0.5, h111,hb110, -0.1, 35, "SSSF_pi_-0.1_h111=0.5")
# SSSF(100, 0.6, h111,hb110, -0.1, 35, "SSSF_pi_-0.1_h111=0.6")
# #
# SSSF(100, 0.2, h001,hb110, -0.1, 35, "SSSF_pi_-0.1_h001=0.2")
# SSSF(100, 0.3, h001,hb110, -0.1, 35, "SSSF_pi_-0.1_h001=0.3")
# SSSF(100, 0.4, h001,hb110, -0.1, 35, "SSSF_pi_-0.1_h001=0.4")

# SSSF(100, 0.8, h1b10,hb110, -0.1, 35, "SSSF_pi_-0.1_h1b10=0.8")
# SSSF(100, 1.0, h1b10,hb110, -0.1, 35, "SSSF_pi_-0.1_h1b10=1.0")
# SSSF(100, 1.2, h1b10,hb110, -0.1, 35, "SSSF_pi_-0.1_h1b10=1.2")

# SSSF(100, 0.8, h110,hb110, -0.1, 35, "SSSF_pi_-0.1_h110=0.8")
# SSSF(100, 1.0, h110,hb110, -0.1, 35, "SSSF_pi_-0.1_h110=1.0")
# SSSF(100, 1.2, h110,hb110, -0.1, 35, "SSSF_pi_-0.1_h110=1.2")
# #
# SSSF(100, 0.2, h111,hb110, 0.02,35, "SSSF_zero_0.02_h111=0.2")
# SSSF(100, 0.3, h111,hb110, 0.02,35, "SSSF_zero_0.02_h111=0.3")
# SSSF(100, 0.4, h111,hb110, 0.02,35, "SSSF_zero_0.02_h111=0.4")

# SSSF(100, 0.2, h001,hb110, 0.02,35, "SSSF_zero_0.02_h001=0.2")
# SSSF(100, 0.3, h001,hb110, 0.02,35, "SSSF_zero_0.02_h001=0.3")
# SSSF(100, 0.4, h001,hb110, 0.02,35, "SSSF_zero_0.02_h001=0.4")
# #
# SSSF(100, 0.6, h110,hb110, 0.02,35, "SSSF_zero_0.02_h110=0.6")
# SSSF(100, 0.8, h110,hb110, 0.02,35, "SSSF_zero_0.02_h110=0.8")
# SSSF(100, 1.0, h110,hb110, 0.02,35, "SSSF_zero_0.02_h110=1.0")

# TWOSPINCON_gang(80, 0, h111, -0.1, 25, "TSC_-0.33_h111=0_gang")
# TWOSPINCON_gang(80, 0.2, h111,  -0.1, 25, "TSC_-0.33_h111=0.2_gang")
# TWOSPINCON_gang(80, 1, h111,  -0.1, 25, "TSC_-0.33_h111=1.0_gang")

# TWOSPINCON_gang(80, 0.2, h001,  -0.1, 25, "TSC_-0.33_h001=0.2_gang")
# TWOSPINCON_gang(80, 1, h001,  -0.1, 25, "TSC_-0.33_h001=1.0_gang")

# TWOSPINCON_gang(80, 0.2, h110,  -0.1, 25, "TSC_-0.33_h110=0.2_gang")
# TWOSPINCON_gang(80, 1, h110,  -0.1, 25, "TSC_-0.33_h110=1.0_gang")



# TWOSPINCON(80, 0, h111,  -0.1, 25, "TSC_-0.33_h111=0")
# TWOSPINCON(80, 0.2, h111,  -0.1, 25, "TSC_-0.33_h111=0.2")
# TWOSPINCON(80, 1, h111,  -0.1, 25, "TSC_-0.33_h111=1.0")

# TWOSPINCON(80, 0.2, h001,  -0.1, 25, "TSC_-0.33_h001=0.2")
# TWOSPINCON(80, 1, h001,  -0.1, 25, "TSC_-0.33_h001=1.0")

# TWOSPINCON(80, 0.2, h110,  -0.1, 25, "TSC_-0.33_h110=0.2")
# TWOSPINCON(80, 1, h110,  -0.1, 25, "TSC_-0.33_h110=1.0")


# TWOSPINCON_gang(80, 0, h111, -0.1, 25, "TSC_test_gang_no_field")
# TWOSPINCON(80, 0, h111, -0.1, 25, "TSC_test_no_field")

# TWOSPINCON_gang(80, 1, h111, 0.001, 25, "TSC_test_gang_ising_h111")
# TWOSPINCON(80, 1, h111,  0.001, 25, "TSC_test_ising_h111")
# TWOSPINCON_gang(80, 1, h001,  0.001, 25, "TSC_test_gang_ising_h001")
# TWOSPINCON(80, 1, h001, 0, 25, "TSC_test_ising_h001")
# TWOSPINCON_gang(80, 1, h110,  0.001, 25, "TSC_test_gang_ising_h110")
# TWOSPINCON(80, 1, h110,  0.001, 25, "TSC_test_ising_h110")


# TWOSPINCON_gang(80, 1, h111, -0.1, 25, "TSC_test_gang")
# TWOSPINCON(80, 1, h111, -0.1, 25, "TSC_test")
 