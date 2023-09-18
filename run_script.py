import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *


# DSSF(0.01, 0, h111,-1/3, "DSSF_-0.33_detailed", 35, 0.02)
# DSSF(0.01, 0, h111,-0.1875, "DSSF_-0.1875_detailed", 35, 0.02)

SSSF(50, 0.2, h111,hb110, -0.1,31, "SSSF_pi_-0.1_h111=0.2")
SSSF(50, 0.3, h111,hb110, -0.1,31, "SSSF_pi_-0.1_h111=0.3")
SSSF(50, 0.4, h111,hb110, -0.1,31, "SSSF_pi_-0.1_h111=0.4")
#
SSSF(50, 0.2, h001,hb110, -0.1,31, "SSSF_pi_-0.1_h001=0.2")
SSSF(50, 0.3, h001,hb110, -0.1,31, "SSSF_pi_-0.1_h001=0.3")
SSSF(50, 0.4, h001,hb110, -0.1,31, "SSSF_pi_-0.1_h001=0.4")

SSSF(50, 0.4, h110,hb110, -0.1,31, "SSSF_pi_-0.1_h110=0.4")
SSSF(50, 0.6, h110,hb110, -0.1,31, "SSSF_pi_-0.1_h110=0.6")
SSSF(50, 0.8, h110,hb110, -0.1,31, "SSSF_pi_-0.1_h110=0.8")
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