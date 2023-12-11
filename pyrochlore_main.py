import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import pyrochlore_dispersion
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import pyrochlore_general as pygen
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
from numpy.testing import assert_almost_equal, assert_allclose

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
# zmag1 = contract('k,ik->i',h1b10,z)
# zmag2 = contract('k,ik->i',np.array([0,-1,1])/np.sqrt(2),z)
# zmag3 = contract('k,ik->i',hb110,z)
# start = time.time()
# C = graphdispersion(-0.01, -0.01, 1, 0, h111, 1, 2, 20, 25, 0)
# plt.show()
Jpm=0.04
B = graphdispersion(-2*Jpm, -2*Jpm, 1, 0, h111, 2, 20, 100, 0)
# C = graphdispersion(-0.01, -0.01, 1, 0, h111, 2, 20, 25, 1)
# D = graphdispersion(0.2, 0.2, 1, 0, h111, 2, 20, 25, 1)
plt.show()
# #
# #
# flux = np.array([1,1,1,1])*np.pi
flux = np.zeros(4)
D = generaldispersion(-2*Jpm, -2*Jpm, 1, 0, h110, 2, 20, 25, flux)
plt.show()

Jxx = -0.08
Jyy = -0.08
Jzz = 1
kappa = 2
BZres = 26
h = 0
n = h111

py0s = py0.zeroFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n)
pyg = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n)
py0s.solvemeanfield()
pyg.solvemeanfield()
py0s.graph(False)
pyg.graph(False)
plt.show()
py0s.MFE()
pyg.MFE()
A = py0s.green_zero(py0s.bigB, py0s.lams)
B = pyg.green_pi(py0s.bigB, py0s.lams)
C = py0.rho_true(py0s.MF, py0s.lams, py0s.Jzz)
D = pygen.rho_true(pyg.Jzz, pyg.MF, pyg.lams)
print(C, D)
# py0s.print_rho()
# pyg.print_rho()
# plt.show()
#
# comparePi(-0.05, 0.05, 25, 0, 0, 0, h110, 26, 2, 'compare')
# compare0(-0.05, 0.05, 25, 0, 0, 0, h110, 26, 2, 'compare0')

# DSSF(0.02, -0.08, -0.08, 1, 0, h111, 'DSSF_general_0_flux', 26, 2)

# sungdispersion(0.2, 0.2, 25, 0, h110, 2, 20, 26)
# plt.show()

# TWOSPINCON_0(100, 0, h111, 0.2, 0.2, 1, 20, 'TSC_0_PSG')

# TWOSPINCON_general(100, 0, h111, 0.2, 0.2, 1, 20, np.zeros(4), 'TSC_0_GENERAL')

# fluxs = np.array([[0, 0, 0, 0],
#                   [np.pi, np.pi, np.pi, np.pi],
#                   [np.pi/4, -3*np.pi/4, np.pi/4, np.pi/4]])

# # PhaseMagtestJP(-0.05, 0.05, 25,  0, 0.3, 25, h110, 26,2, '0_1')
# generalJPSweep(-0.05, 0.05, 25, 0, h110, 26, 2, fluxs, '01')
# generalHSweep(0, 0, 0.3, 25, h111, 26, 2, fluxs, 'test_111_1')
# #
# fluxs = np.array([[0, 0, 0, 0],
#                   [np.pi, np.pi, np.pi, np.pi],
#                   [np.pi, np.pi, 0, 0]])

# generalJPSweep(-0.05, 0.05, 25, 0, h110, 26, 2, fluxs, '0_pp00')
# generalHSweep(0, 0, 0.3, 25, h110, 26, 2, fluxs, 'test_110_1')
# PhaseMagtestH(0, 0, 0,  0, 0.3, 25, h110, 26,2, 'test1_110_1')

# findXYZPhase(-1, 1, 5, 26, 2, "XYZ_0_field")
