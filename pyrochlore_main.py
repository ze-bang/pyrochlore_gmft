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


getminflux110("Niagara_data_3/h110_flux_plane_zero_0.2")
getminflux110("Niagara_data_3/h110_flux_plane_pi_0.2")
getminflux110("Niagara_data_3/h110_flux_plane_mid_0.2")

getminflux111("Niagara_data_3/h111_flux_plane_zero_0.2")
getminflux111("Niagara_data_3/h111_flux_plane_pi_0.2")
getminflux111("Niagara_data_3/h111_flux_plane_mid_0.2")

n = 20
h = 0.3
BZres = 40
JP = np.linspace(0,1, n)
ppp0f = generateflux110(0, np.pi, 0, 0)
pp00f = generateflux110(np.pi, 0, 1, 0)

MFE0 = np.zeros(n)
MFEpi = np.zeros(n)
MFEppp0 = np.zeros(n)
MFEpp00 = np.zeros(n)

for i in range(n):
    A = pygen.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=np.zeros(4))
    B = pygen.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=ppp0f)
    D = pygen.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=pp00f)
    C = pygen.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=np.array([np.pi, np.pi, np.pi, np.pi]))

    A.solvemeanfield()
    B.solvemeanfield()
    D.solvemeanfield()
    C.solvemeanfield()

    MFE0[i] = A.MFE()
    MFEppp0[i] = B.MFE()
    MFEpp00[i] = D.MFE()
    MFEpi[i] = C.MFE()

    print(JP[i], MFE0[i], A.qmin, MFEppp0[i], B.qmin, MFEpp00[i], D.qmin, MFEpi[i], C.qmin)

plt.plot(JP, MFE0, label='0')
plt.plot(JP, MFEpp00, label=r'$\pi\pi 0 0$')
plt.plot(JP, MFEppp0, label=r'$\pi\pi \pi 0$')
plt.plot(JP, MFEpi, label=r'$\pi$')
# plt.plot(JP, MFE0old, label='0 old')
# plt.plot(JP, MFEpiold, label=r'$\pi$ old')
plt.legend()
plt.show()
# flux = np.array([np.pi, np.pi, 0, 0])
# findPhaseMag(-0.5, 0.1, 25, 0, 0.3, 25, h110, 2, 2, flux, 'test')


# k = genBZ(25)
# zmag = contract('k,ik->i', h111, z)
# ffact = contract('ik, jk->ij', k, NN)
# ffact = np.exp(1j * ffact)

# flux = np.array([np.pi, np.pi, 0, 0])
# # flux = generateflux110(0, 0, 1, 1)
# flux = generateflux111(-np.pi/4, np.pi/4,1)
# print(flux)
# A = pygen.piFluxSolver(0, 0, 1, kappa=2, graphres=graphres, BZres=25, h=0, n=h110, flux=flux)
# E0 = A.A_pi_here
# B0 = A.A_pi_rs_traced_here
# M0 = contract('ku, u, ru, urx->krx', ffact, zmag, np.exp(1j*E0), piunitcell)
# D = generaldispersion(0.2, 0.2, 1, 0, h111, 2, 20, 25, flux)
# plt.show()

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