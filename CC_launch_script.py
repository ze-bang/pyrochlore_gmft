import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *



# SSSF(100, 0.2, h001,hb110, -1/3,50, "SSSF_pi_-0.33_h001=0.2")
SSSF(100, 0, h001, hb110, -0.05, 50, "SSSF_pi_-0.05_DETAILED")
SSSF(100, 0, h111, hb110, -0.20, 50, "SSSF_pi_-0.20_DETAILED")
SSSF(100, 0, h111, hb110, -0.40, 50, "SSSF_pi_-0.40_DETAILED")



SSSF(100, 0, h111, hb110, 0.02, 50, "SSSF_zero_0.02_DETAILED")
SSSF(100, 0, h111, hb110, 0.03, 50, "SSSF_zero_0.03_DETAILED")
SSSF(100, 0, h111, hb110, 0.04, 50, "SSSF_zero_0.04_DETAILED")



# SSSF(50, 0, np.array([1,1,1]),-0.40,30, "SSSF_pi_-0.40_DETAILED")

# SSSF(25, 0, h111,hb110, 0.04,10, "SSSF_zero_0.02")

SSSF(100, 0.2, h111,hb110, -1/3,50, "SSSF_pi_-0.33_h111=0.2")
SSSF(100, 0.3, h111,hb110, -1/3,50, "SSSF_pi_-0.33_h111=0.3")
SSSF(100, 0.4, h111,hb110, -1/3,50, "SSSF_pi_-0.33_h111=0.4")
#
SSSF(100, 0.2, h001,hb110, -1/3,50, "SSSF_pi_-0.33_h001=0.2")
SSSF(100, 0.3, h001,hb110, -1/3,50, "SSSF_pi_-0.33_h001=0.3")
SSSF(100, 0.4, h001,hb110, -1/3,50, "SSSF_pi_-0.33_h001=0.4")

SSSF(100, 0.4, h110,hb110, -1/3,50, "SSSF_pi_-0.33_h110=0.4")
SSSF(100, 0.6, h110,hb110, -1/3,50, "SSSF_pi_-0.33_h110=0.6")
SSSF(100, 0.8, h110,hb110, -1/3,50, "SSSF_pi_-0.33_h110=0.8")
#
SSSF(100, 0.2, h111,hb110, 0.02,50, "SSSF_zero_0.02_h111=0.2")
SSSF(100, 0.3, h111,hb110, 0.02,50, "SSSF_zero_0.02_h111=0.3")
SSSF(100, 0.4, h111,hb110, 0.02,50, "SSSF_zero_0.02_h111=0.4")

SSSF(100, 0.2, h001,hb110, 0.02,50, "SSSF_zero_0.02_h001=0.2")
SSSF(100, 0.3, h001,hb110, 0.02,50, "SSSF_zero_0.02_h001=0.3")
SSSF(100, 0.4, h001,hb110, 0.02,50, "SSSF_zero_0.02_h001=0.4")
#
SSSF(100, 0.4, h110,hb110, 0.02,50, "SSSF_zero_0.02_h110=0.4")
SSSF(100, 0.6, h110,hb110, 0.02,50, "SSSF_zero_0.02_h110=0.6")
SSSF(100, 0.8, h110,hb110, 0.02,50, "SSSF_zero_0.02_h110=0.8")

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
 