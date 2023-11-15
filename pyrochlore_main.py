import pyrochlore_dispersion
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
import netCDF4 as nc
import warnings


# JP = np.linspace(-0.5, 0.1, 400)
# h = np.linspace(0, 1, 100)
#
# filename="phase_111_kappa=2_complete"
#
# rectemp1 = np.loadtxt("CC_Files/phase_111_kappa=2_complete_gap.txt")
# rectemp = np.loadtxt("CC_Files/phase_111_kappa=2_complete.txt")
#
# plt.contourf(JP, h, rectemp.T)
#
# plt.show()

# graphMagPhase(JP, h, rectemp1,'Files/' + filename + '_gap')
# graphMagPhase(JP, h, rectemp,'Files/' + filename)


#region graph dispersion

#endregion


# graphedges(-0.1, 0.2, h001, 1, 1, 20, 20, True)
# graphedges(-0.1, 0.2, h1b10, 1, 1, 20, 20, True)

# graphdispersion(0.1, 1, h1b10, 1, 2, 20, 20)
# graphdispersion(1, 1, 1, 2, h1b10, 1, 2, 20, 20, False)
# plt.show()
zmag1 = contract('k,ik->i',h1b10,z)
zmag2 = contract('k,ik->i',np.array([0,-1,1])/np.sqrt(2),z)
# zmag3 = contract('k,ik->i',hb110,z)
# C = graphdispersion(0, 0, 1, 0.2, h1b10, 1, 2, 20, 25, True)
# plt.show()

# D = graphdispersion(0, 0, 1, 0.2, np.array([0,-1,1])/np.sqrt(2), 1, 2, 20, 25, True)
# E = graphdispersion(0, 0, 1, 0.2, np.array([0,1,1])/np.sqrt(2), 1, 2, 20, 25, True)
# plt.show()

# C = graphdispersion(-0.08, -0.08, 1, 3, h1b10, 1, 2, 20, 25)
# C = graphdispersion(-0.08, -0.08, 1, 3, h1b10, 1, 2, 20, 26)
# C = graphdispersion(-0.08, -0.08, 1, 3, h1b10, 1, 2, 20, 27)
# C = graphdispersion(-0.08, -0.08, 1, 3, h1b10, 1, 2, 20, 25)
# C = graphdispersion(-0.08, -0.08, 1, 3, h1b10, 1, 2, 20, 29)
# C = graphdispersion(-0.08, -0.08, 1, 3, h1b10, 1, 2, 20, 30)


# C = graphdispersion(0.2, 0.2, 1, 3, h1b10, 1, 2, 20, 55)
# plt.show()
# D = graphdispersion(0.5, 0.5, 1, 0.1, h1b10, 1, 2, 20, 20, True)

# graphdispersion_old(-0.1, 0.2, h1b10, 1, 2, 20, 20)
# graphdispersion(-0.1, 1, h001, 1, 2, 20, 20)
# M = pypi.M_pi_mag_sub_single(np.pi*np.array([0.1,0.2,0.3]), 1, np.array([0,0,1]))
# graphdispersion(-0.1, 0.2, h001, 1, 2, 20, 20)
#
# graphdispersion(-0.1, 1, h111, 1, 2, 20, 20)
# graphdispersion(-0.1, 1, h110, 1, 2, 20, 20)
# graphdispersion(-0.1, 1, h001, 1, 2, 20, 20)

# graphdispersion(0.046, 0.2, h111, 1, 2, 20, 20)
# graphdispersion_wrong(-1/3, 0, h111, 1, 1, 20, 20)
# graphdispersion(0.046,0, h111, 1, 2, 20, 20)
# graphdispersion(0.046, 0, h111, 1, 2, 20, 20)
# graphdispersion(0.05, 0, h111, 1, 2, 20, 20)
# graphdispersion(0.1, 0, h111, 1, 2, 20, 20)
# plt.show()

# findPhase(60,20, 20, "Files/phase_diagram.txt")

PhaseMagtestH(0, 0.25, 25, 0, 0.3, 50, h110, 30, 2, "ising_h110")
PhaseMagtestH(0, 0.25, 25, 0, 0.3, 50, h001, 30, 2, "ising_h001")
PhaseMagtestH(0, 0.25, 25, 0, 0.3, 50, h111, 30, 2, "ising_h111")
PhaseMagtestH(0, 0.25, 25, 0, 0.3, 50, h1b10, 30, 2, "ising_h1b10")
PhaseMagtestH(0, 0.25, 25, 0, 1, 50, h110, 30, 2, "ising_h110_zoomout")
PhaseMagtestH(0, 0.25, 25, 0, 1, 50, h001, 30, 2, "ising_h001_zoomout")
PhaseMagtestH(0, 0.25, 25, 0, 1, 50, h111, 30, 2, "ising_h111_zoomout")
PhaseMagtestH(0, 0.25, 25, 0, 1, 50, h1b10, 30, 2, "ising_h1b10_zoomout")


# PhaseMagtestH(0, 0.25, 25, 0, 0.3, 50, h100, 30, 2, "ising_h100")
# PhaseMagtestH(0, 0.25, 25, 0, 1, 50, h100, 30, 2, "ising_h100_zoomout")
#
# PhaseMagtestHGS(0, 0.25, 25, 0, 1, 50, h110, 30, 1, "ising_h110_kappa=1_zoomout")
# PhaseMagtestHGS(0, 0.25, 25, 0, 1, 50, h001, 30, 1, "ising_h001_kappa=1_zoomout")
# PhaseMagtestHGS(0, 0.25, 25, 0, 1, 50, h111, 30, 1, "ising_h111_kappa=1_zoomout")
# PhaseMagtestHGS(0, 0.25, 25, 0, 1, 50, h1b10, 30, 1, "ising_h1b10_kappa=1_zoomout")

# PhaseMagtestJP(-0.1, 0.1, 25, 0, 3, 25, h001, 35, 2, "0.png")
#
#


# PhaseMagtestH(0, 0.1, 25, 0, 3, 25, h001, 25, 2, "0.25.png")


# PhaseMagtestJP(-0.2, 0.2, 25, 0.1, 3, 25, h1b10, 25, 2, "0.1_h1b10.png")
# PhaseMagtestJP(0.045, 0.0505, 30, 0.2, 3, 30, h110, 25, 2, "0.2_h110.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.3, 3, 25, h110, 25, 2, "0.3_h110.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.4, 3, 25, h110, 25, 2, "0.4_h110.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.5, 3, 25, h110, 25, 2, "0.5_h110.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.6, 3, 25, h110, 25, 2, "0.6_h110.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.7, 3, 25, h110, 25, 2, "0.7_h110.png")
#
# PhaseMagtestJP(-0.2, 0.2, 25, 0.1, 3, 25, h111, 25, 2, "0.1_h111.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.2, 3, 25, h111, 25, 2, "0.2_h111.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.3, 3, 25, h111, 25, 2, "0.3_h111.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.4, 3, 25, h111, 25, 2, "0.4_h111.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.5, 3, 25, h111, 25, 2, "0.5_h111.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.6, 3, 25, h111, 25, 2, "0.6_h111.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.7, 3, 25, h111, 25, 2, "0.7_h111.png")
#
# PhaseMagtestJP(-0.2, 0.2, 25, 0.1, 3, 25, h001, 25, 2, "0.1_h001.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.2, 3, 25, h001, 25, 2, "0.2_h001.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.3, 3, 25, h001, 25, 2, "0.3_h001.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.4, 3, 25, h001, 25, 2, "0.4_h001.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.5, 3, 25, h001, 25, 2, "0.5_h001.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.6, 3, 25, h001, 25, 2, "0.6_h001.png")
# PhaseMagtestJP(-0.2, 0.2, 25, 0.7, 3, 25, h001, 25, 2, "0.7_h001.png")
#

# PhaseMagtestJP(-0.2, 0.2, 25, 0, 0, 10, h1b10, 26, 2, "0.png")

# PhaseMagtestH(0, 0.5, 50, 2, 4, 25, h1b10, 30, 1, "4_kappa=1.png")
# PhaseMagtestJP(0.01, 0.3, 25, 2, 4, 25, h1b10, 26, 2, "4_1.png")
# MagJP(0.0499, 0.0501, 25, 0, 4, 25, h1b10, 26, 2, "Sx_kappa=1.png")
# PhaseMagtestJP(-0.2, 0.1, 25, 0.23, 4, 25, h1b10, 26, 2, "0.23_h1b10_1.png")
# PhaseMagtestJP(-0.2, 0.1, 25, 0.26, 4, 25, h1b10, 26, 2, "0.26_h1b10_1.png")
# PhaseMagtestJP(-0.2, 0.1, 25, 0.24, 4, 25, h1b10, 26, 2, "0.24_h1b10_1.png")
# PhaseMagtestJP(-0.1, 0.1, 25, 0.22, 3, 25, h001, 35, 2, "0.22.png")
# PhaseMagtestJP(-0.1, 0.1, 25, 0.23, 3, 25, h001, 35, 2, "0.23.png")
# PhaseMagtestJP(-0.1, 0.1, 25, 0.24, 3, 25, h001, 35, 2, "0.24.png")


# samplegraph(100, ["SSSF_pi_0.02_DETAILED", "SSSF_pi_0.03_DETAILED", "SSSF_pi_0.04_DETAILED"])

# SSSF(100, 0.2, 0.2, 1, 0, h001, hb110, 25, "SSSF_pi_-0.05_DETAILED", True)
# SSSF(100, 0, h111, hb110, -0.20, 50, "SSSF_pi_-0.20_DETAILED")
# SSSF(100, 0, h111, hb110, -0.40, 50, "SSSF_pi_-0.40_DETAILED")
#
#
#
# SSSF(25, 0, h111, hb110, 0.5, 0.5, 1, 25, "SSSF_test")
# SSSF(25, 0, h111, hb110, 0.5, 0.5, 1, 25, "SSSF_test_new", True)
# SSSF(100, 0, h111, hb110, 0.03, 50, "SSSF_pi_0.03_DETAILED")
# SSSF(100, 0, h111, hb110, 0.04, 50, "SSSF_pi_0.04_DETAILED")



# SSSF(50, 0, np.array([1,1,1]),-0.40,30, "SSSF_pi_-0.40_DETAILED")

# SSSF(25, 0, h111,hb110, 0.04,10, "SSSF_zero_0.02")

# SSSF(100, 0.2, h111,hb110, -1/3,50, "SSSF_pi_-0.33_h111=0.2")
# SSSF(100, 0.3, h111,hb110, -1/3,50, "SSSF_pi_-0.33_h111=0.3")
# SSSF(100, 0.4, h111,hb110, -1/3,50, "SSSF_pi_-0.33_h111=0.4")
#
# SSSF(40, 0.2, h001,hb110, -1/3,25, "SSSF_pi_-0.33_h001=0.2")
# SSSF(100, 0.3, h001,hb110, -1/3,50, "SSSF_pi_-0.33_h001=0.3")
# SSSF(100, 0.4, h001,hb110, -1/3,50, "SSSF_pi_-0.33_h001=0.4")
#
# SSSF(100, 0.4, h110,hb110, -1/3,50, "SSSF_pi_-0.33_h110=0.4")
# SSSF(100, 0.6, h110,hb110, -1/3,50, "SSSF_pi_-0.33_h110=0.6")
# SSSF(100, 0.8, h110,hb110, -1/3,50, "SSSF_pi_-0.33_h110=0.8")
# #
# SSSF(100, 0.2, h111,hb110, 0.02,50, "SSSF_zero_0.02_h111=0.2")
# SSSF(100, 0.3, h111,hb110, 0.02,50, "SSSF_zero_0.02_h111=0.3")
# SSSF(100, 0.4, h111,hb110, 0.02,50, "SSSF_zero_0.02_h111=0.4")
#
# SSSF(100, 0.2, h001,hb110, 0.02,50, "SSSF_zero_0.02_h001=0.2")
# SSSF(100, 0.3, h001,hb110, 0.02,50, "SSSF_zero_0.02_h001=0.3")
# SSSF(100, 0.4, h001,hb110, 0.02,50, "SSSF_zero_0.02_h001=0.4")
# #
# SSSF(100, 0.4, h110,hb110, 0.02,50, "SSSF_zero_0.02_h110=0.4")
# SSSF(100, 0.6, h110,hb110, 0.02,50, "SSSF_zero_0.02_h110=0.6")
# SSSF(100, 0.8, h110,hb110, 0.02,50, "SSSF_zero_0.02_h110=0.8")

# DSSF(0.01, 0, h111,-0.1875, "DSSF_-0.1875_detailed", 35, 0.02)



#
# graphPhase("Files/phase_diagram.txt")

# leng = len(np.concatenate((genBZ(25), symK)))
#
# with nc.Dataset("Files/phase_test_111_kappa=2_q_condensed.nc", "r") as dataset:
#
#     temp_var = dataset.variables["q_condensed"][:]
#
# print(temp_var)

# findPhaseMag(-0.3, 0.1, 25, 0, 1, 25, h111, 25, 2, "phase_test_111_kappa=2")
# findPhaseMag(-0.3, 0.1, 25, 0, 1, 25, h001, 25, 2, "phase_test_001_kappa=2")
# findPhaseMag(-0.5, 0.3, 50, 0, 4, 25, h110, 25, 2, "phase_test_110_kappa=2")
# findPhaseMag(-0.3, 0.3, 30, 0, 1, 25, h111, 25, 2, "phase_test_111_kappa=2")
# findPhaseMag(-0.5, 0.1, 300, 0, 1, 100, h001, 25, 2, "phase_test_001_kappa=2")
# findPhaseMag(-0.5, 0, 300, 0, 4, 100, h110, 25, 2, "phase_test_110_kappa=2")
# findPhaseMag(-0.5, 0.1, 300, 0, 1, 150, h111, 25, 2, "phase_111_kappa=2_complete")
# findPhaseMag(-0.5, 0.1, 300, 0, 1, 150, h001, 25, 2, "phase_001_kappa=2_complete")
# findPhaseMag(-0.5, 0.1, 300, 0, 4, 150, h110, 25, 2, "phase_110_kappa=2_complete")

# findPhaseMag(-0.5, 0.1, 300, 0, 1, 150, h001, 25, 2, "phase_001_kappa=2_complete")

# findPhaseMag_pi_zero(-0.5, 0, 300, 0, 1, 150, h111, 25, 2, "phase_111_kappa=2_complete")
# findPhaseMag_pi_zero(-0.5, 0, 300, 0, 1, 150, h001, 25, 2, "phase_001_kappa=2_complete")
# findPhaseMag_pi_zero(-0.5, 0, 300, 0, 4, 150, h110, 25, 2, "phase_110_kappa=2_complete")

# #
# DSSF(0.01,0, h111, 0.04, "DSSF_0.04_detailed", 35, 0.02)


# TWOSPINCON(40, 0.4, h111, -1/3, 25, "TSC_-0.33_h111=0")
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

# print(True*2)

# PhaseMagtestJP(-0.05, 0.05, 10, 0, 0, 0, h111, 20, 2, "test")

# findPhaseMag(-0.1, 0.1, 100, 0, 1, 100, h111, 5, 2, "phase_test_111_kappa=2_0_flux")
# findPhaseMag(0, 0.1, 100, 0, 1, 100, h001, 25, 2, "phase_test_001_kappa=2_0_flux")
# findPhaseMag(0, 0.1, 100, 0, 4, 100, h110, 25, 2, "phase_test_110_kappa=2_0_flux")


# graphMagPhase("phase_test_111_kappa=1", 0.25,3)
# graphMagPhase("phase_test_001_kappa=1", 0.25,3)
# graphMagPhase("phase_test_110_kappa=1", 0.25,12)

# graphMagPhase("phase_test_001", 0.25,3)
# graphMagPhase("phase_test_110", 0.25,12)