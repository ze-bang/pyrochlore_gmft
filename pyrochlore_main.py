import os

import numpy as np

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import warnings
import pyrochlore_conclusive as pycon
from variation_flux import *


# getminflux110("Niagara_data_3/h110_flux_plane_zero_0.2")
# getminflux110("Niagara_data_3/h110_flux_plane_pi_0.2")
# getminflux110("Niagara_data_3/h110_flux_plane_mid_0.2")
#
# getminflux111("Niagara_data_3/h111_flux_plane_zero_0.2")
# getminflux111("Niagara_data_3/h111_flux_plane_pi_0.2")
# getminflux111("Niagara_data_3/h111_flux_plane_mid_0.2")

# JP = 0.02
# h = 0.3
# BZres = 20
#
# z0 = pycon.piFluxSolver(-2 * JP, -2 * JP, 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=np.zeros(4))
# z0.solvemeanfield()
# print(z0.condensed, z0.lams, z0.minLams, z0.qmin, z0.GS(), z0.MFE(), z0.rhos, z0.delta, z0.chi, z0.chi0, z0.xi, z0.magnetization())
# z0.graph(False)
# #
#
# N = 20
# lams = np.zeros(N)
# offset=60
# for i in range(offset,offset+N):
#     z0 = pygen.piFluxSolver(-2 * JP, -2 * JP, 1, kappa=2, graphres=graphres, BZres=i+1, h=h, n=h110, flux=np.zeros(4))
#     z0.solvemeanfield()
#     lams[i-offset] = z0.lams[0]
#     print(i+1,lams[i-offset])
#
# plt.plot(lams)


n = 31
h = 0.3
JP = -0.01

BZgrid = np.linspace(10,9+n-1, n-1, dtype=int)

GSA = np.zeros(n)
GSB = np.zeros(n)
GSC = np.zeros(n)
GSD = np.zeros(n)
GSE = np.zeros(n)

# mathematicapi = 0.220122539071829
mathematica0 = 0.2201538131708155

#mathematica0_h_0.3_jp_0.005=0.221507896850163
#mathematicapi_h_0.3_jp_0.005=0.221502970982931
#mathematicapp00_h_0.3_jp_0.005=0.221515468258903

fluxtotest = np.ones(4)*np.pi

for i in range(n):
    A = pycon.piFluxSolver(-2*JP, -2*JP, 1, kappa=2, graphres=graphres, BZres=i+10, h=h, n=h110, flux=fluxtotest, intmethod=gauss_quadrature_3D_pts)
    # B = pycon.piFluxSolver(-2*JP, -2*JP, 1, kappa=2, graphres=graphres, BZres=i+10, h=h, n=h110, flux=fluxtotest, intmethod=riemann_sum_3d_pts)
    C = pycon.piFluxSolver(-2*JP, -2*JP, 1, kappa=2, graphres=graphres, BZres=i+10, h=h, n=h110, flux=fluxtotest, intmethod=trapezoidal_rule_3d_pts)
    # D = pycon.piFluxSolver(-2*JP, -2*JP, 1, kappa=2, graphres=graphres, BZres=i+10, h=h, n=h110, flux=fluxtotest, intmethod=monte_carlo_integration_3d_pts)
    E = pycon.piFluxSolver(-2*JP, -2*JP, 1, kappa=2, graphres=graphres, BZres=i+10, h=h, n=h110, flux=fluxtotest, intmethod=simpsons_rule_3d_pts)

    # C = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=i+1, h=h, n=h110, flux=np.array([np.pi, np.pi, np.pi, np.pi]))

    A.solvemeanfield()
    # B.solvemeanfield()
    C.solvemeanfield()
    # D.solvemeanfield()
    E.solvemeanfield()

    GSA[i] = A.GS()
    # GSB[i] = B.GS()
    GSC[i] = C.GS()
    # GSD[i] = D.GS()
    GSE[i] = E.GS()

    print(i+10, GSA[i], A.lams[0], GSB[i], GSC[i], C.lams[0], GSD[i], GSE[i], E.lams[0])

# A = np.loadtxt("temp.txt", unpack=True)
#
# GSA = A[1]
# GSC = A[4]
# GSE = A[7]
plt.plot(BZgrid,np.log(np.abs(np.diff(GSA))), label='Gauss Quadrature')
# plt.plot(BZgrid,GSB-mathematicapi, label='Riemann')
plt.plot(BZgrid,np.log(np.abs(np.diff(GSC))), label='Trapezoidal')
# plt.plot(BZgrid,GSD-mathematicapi, label='Monte Carlo')
plt.plot(BZgrid,np.log(np.abs(np.diff(GSE))), label='Simpsons Rule')


# plt.plot(JP, MFE0old, label='0 old')
# plt.plot(JP, MFEpiold, label=r'$\pi$ old')
plt.legend()
plt.show()

# n = 40
# h = 0.2
# BZres = 40
# JP = np.linspace(-0.03,0.01, n)
#
# GS0 = np.zeros(n)
# GSpi = np.zeros(n)
# GSpp00 = np.zeros(n)
#
# MFE0 = np.zeros(n)
# MFEpi = np.zeros(n)
# MFEpp00 = np.zeros(n)
# # warnings.filterwarnings("error")
#
# for i in range(n):
#     A = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=np.zeros(4))
#     D = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=pp00f)
#     C = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=np.array([np.pi, np.pi, np.pi, np.pi]))
#
#     A.solvemeanfield()
#     D.solvemeanfield()
#     C.solvemeanfield()
#
#
#     GS0[i] = A.GS()
#     GSpp00[i] = D.GS()
#     GSpi[i] = C.GS()
#
#
#     print(JP[i], GS0[i], A.qmin, GSpp00[i], D.qmin, GSpi[i], C.qmin)

# plt.plot(JP, MFE0, label='0')
# plt.plot(JP, MFEpp00, label=r'$\pi\pi 0 0$')
# plt.plot(JP, MFEpi, label=r'$\pi$')
# # plt.plot(JP, MFE0old, label='0 old')
# # plt.plot(JP, MFEpiold, label=r'$\pi$ old')
# plt.legend()
# plt.show()

# plt.plot(JP, GS0, label='0')
# plt.plot(JP, GSpp00, label=r'$\pi\pi 0 0$')
# plt.plot(JP, GSpi, label=r'$\pi$')
# # plt.plot(JP, MFE0old, label='0 old')
# # plt.plot(JP, MFEpiold, label=r'$\pi$ old')
# plt.legend()
# plt.show()
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