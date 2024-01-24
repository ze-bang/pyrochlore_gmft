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
from variation_flux import *
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

# n = 20
# JP = np.linspace(0,0.05, n)
# MFE0 = np.zeros(n)
# MFEpp00 = np.zeros(n)
# MFE0old = np.zeros(n)
# MFEpiold = np.zeros(n)
# for i in range(n):
#     # A = py0.zeroFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=120, h=0.3, n=h110)
#     B = pygen.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=35, h=0.3, n=h110, flux=np.array([np.pi, np.pi, np.pi, np.pi]), A00=np.pi/6, A01=np.pi/6, A02=np.pi/6, A10=0)
#     # C = pygen.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=35, h=0.3, n=h110, flux=np.array([np.pi, np.pi, np.pi, np.pi]), A00=0, A01=0, A02=0, A10=0)
#     D = pypi.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=35, h=0.3, n=h110)
#     # A.solvemeanfield()
#     B.solvemeanfield()
#     # C.solvemeanfield()
#     D.solvemeanfield()
#     # MFE0old[i] = A.MFE()
#     MFEpp00[i] = B.MFE()
#     # MFE0[i] = C.MFE()
#     MFEpiold[i] = D.MFE()
#     print(JP[i], MFEpp00[i], MFEpiold[i])
#
# # plt.plot(JP, MFE0old, label='0 Flux old')
# plt.plot(JP, MFEpp00, label='pi Flux')
# # plt.plot(JP, MFE0, label='pi Flux weird')
# plt.plot(JP, MFEpiold, label='pi Flux old')
# plt.legend()
# plt.show()
# flux = np.array([np.pi, np.pi, 0, 0])
# findPhaseMag(-0.5, 0.1, 25, 0, 0.3, 25, h110, 2, 2, flux, 'test')


# k = genBZ(25)
# zmag = contract('k,ik->i', h111, z)
# ffact = contract('ik, jk->ij', k, NN)
# ffact = np.exp(1j * ffact)

flux = np.array([np.pi, np.pi, 0, 0])
flux = np.array([1,2,3,4])
# A = pygen.piFluxSolver(0, 0, 1, kappa=2, graphres=graphres, BZres=25, h=0, n=h110, flux=flux)
# E0 = A.A_pi_here
# B0 = A.A_pi_rs_traced_here
# M0 = contract('ku, u, ru, urx->krx', ffact, zmag, np.exp(1j*E0), piunitcell)
D = generaldispersion(0.2, 0.2, 1, 0, h110, 2, 20, 25, flux)
plt.show()

# graphdispersion(0.2, 0.2, 1, 0, h110, 2, 20, 25, 1)
# plt.show()


# flux = np.array([2,2,2,2])*np.pi
# B = pygen.piFluxSolver(0, 0, 1, kappa=2, graphres=graphres, BZres=25, h=0, n=h110, flux=flux)
# E1 = B.A_pi_here
# B1 = B.A_pi_rs_traced_here
# M1 = contract('ku, u, ru, urx->krx', ffact, zmag, np.exp(1j*E1), piunitcell)
# # D = generaldispersion(-0.08, -0.08, 1, 0.3, h110, 2, 20, 25, flux)
# # plt.show()

# flux = np.array([2,2,0,0])*np.pi
# C = pygen.piFluxSolver(0, 0, 1, kappa=2, graphres=graphres, BZres=25, h=0, n=h110, flux=flux)
# E2 = C.A_pi_here
# B2 = C.A_pi_rs_traced_here
# M2 = contract('ku, u, ru, urx->krx', ffact, zmag, np.exp(1j*E2), piunitcell)
# # D = generaldispersion(-0.08, -0.08, 1, 0.3, h110, 2, 20, 25, flux)
# # plt.show()

# flux = np.array([2,0,0,0])*np.pi
# C = pygen.piFluxSolver(0, 0, 1, kappa=2, graphres=graphres, BZres=25, h=0, n=h110, flux=flux)
# E3 = C.A_pi_here
# B3 = C.A_pi_rs_traced_here
# M3 = contract('ku, u, ru, urx->krx', ffact, zmag, np.exp(1j*E3), piunitcell)
# # D = generaldispersion(-0.08, -0.08, 1, 0.3, h110, 2, 20, 25, flux)
# # plt.show()

# print()
# D = generaldispersion(-0.08, -0.08, 1, 0.3, h110, 2, 20, 25, flux)

# comparePi(-0.05, 0.05, 25, 0, 0, 0, h110, 26, 2, 'compare')
# compare0(-0.05, 0.05, 25, 0, 0, 0, h110, 26, 2, 'compare0_1')
# checkConvergence(0.03, 0, h110, 1, 50, 50, 2, 'check_conv')

# DSSF(0.02, -0.08, -0.08, 1, 0, h111, 'DSSF_general_0_flux', 26, 2)

# plot_MFE_flux(-0.001, -0.001, 1, 0.3, h110, 2, 26, 25)