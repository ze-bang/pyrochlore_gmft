import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import warnings
import pyrochlore_conclusive as pycon
from variation_flux import *
from phase_diagram import *

# plotLinefromnetCDF(h100, h=0.18, diff=True)
# plt.show()

# A = netCDF4.Dataset("Nia_Full_Scan/HanYan_100_Jpm_-0.1_0.1_h_0_0.3_0_flux_full_info.nc")
#
# E = np.array(A.variables['q_condensed'][:])
# print()


Jpm = -0.3
h = 0.5


# flux = np.ones(4)*np.pi
#
flux = np.zeros(4)
# # flux = np.array([np.pi,0,0,np.pi])
# # flux = np.array([0,np.pi,np.pi,0])
# # flux = np.array([np.pi,np.pi,0,0])
# # flux = np.array([0,0,np.pi,np.pi])
#
py0s = pycon.piFluxSolver(-2*Jpm, -2*Jpm, 1, kappa=2, graphres=graphres, BZres=25, h=h, n=h111, flux=flux)
py0s.solvemeanfield()
# M = py0s.M_true(py0s.qmin)
# E, V = np.linalg.eigh(M)
print(py0s.condensed,py0s.qminT, py0s.qmin, py0s.MFE())
ffact = contract('ik, jlk->i', py0s.qmin, NNminus)
ffactA = np.exp(-1j * ffact)
ffactB = np.exp(1j * ffact)
print(ffactA, ffactB)
# #
# generaldispersion(Jpm, Jpm, 1, h, h100,2,20,25,np.array([np.pi,0,0,np.pi]))
# plt.show()
# # n = 31
# h = 0.3
# JP = -0.01
#
# BZgrid = np.linspace(10,9+n-1, n-1, dtype=int)
#
# GSA = np.zeros(n)
# GSB = np.zeros(n)
# GSC = np.zeros(n)
# GSD = np.zeros(n)
# GSE = np.zeros(n)
#
# # mathematicapi = 0.220122539071829
# mathematica0 = 0.2201538131708155
#
# #mathematica0_h_0.3_jp_0.005=0.221507896850163
# #mathematicapi_h_0.3_jp_0.005=0.221502970982931
# #mathematicapp00_h_0.3_jp_0.005=0.221515468258903
#
# fluxtotest = np.ones(4)*np.pi
#
# for i in range(n):
#     A = pycon.piFluxSolver(-2*JP, -2*JP, 1, kappa=2, graphres=graphres, BZres=i+10, h=h, n=h110, flux=fluxtotest, intmethod=gauss_quadrature_3D_pts)
#     # B = pycon.piFluxSolver(-2*JP, -2*JP, 1, kappa=2, graphres=graphres, BZres=i+10, h=h, n=h110, flux=fluxtotest, intmethod=riemann_sum_3d_pts)
#     C = pycon.piFluxSolver(-2*JP, -2*JP, 1, kappa=2, graphres=graphres, BZres=i+10, h=h, n=h110, flux=fluxtotest, intmethod=trapezoidal_rule_3d_pts)
#     # D = pycon.piFluxSolver(-2*JP, -2*JP, 1, kappa=2, graphres=graphres, BZres=i+10, h=h, n=h110, flux=fluxtotest, intmethod=monte_carlo_integration_3d_pts)
#     E = pycon.piFluxSolver(-2*JP, -2*JP, 1, kappa=2, graphres=graphres, BZres=i+10, h=h, n=h110, flux=fluxtotest, intmethod=simpsons_rule_3d_pts)
#
#     # C = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=i+1, h=h, n=h110, flux=np.array([np.pi, np.pi, np.pi, np.pi]))
#
#     A.solvemeanfield()
#     # B.solvemeanfield()
#     C.solvemeanfield()
#     # D.solvemeanfield()
#     E.solvemeanfield()
#
#     GSA[i] = A.GS()
#     # GSB[i] = B.GS()
#     GSC[i] = C.GS()
#     # GSD[i] = D.GS()
#     GSE[i] = E.GS()
#
#     print(i+10, GSA[i], A.lams[0], GSB[i], GSC[i], C.lams[0], GSD[i], GSE[i], E.lams[0])
#
# # A = np.loadtxt("temp.txt", unpack=True)
# #
# # GSA = A[1]
# # GSC = A[4]
# # GSE = A[7]
# plt.plot(BZgrid,np.log(np.abs(np.diff(GSA))), label='Gauss Quadrature')
# # plt.plot(BZgrid,GSB-mathematicapi, label='Riemann')
# plt.plot(BZgrid,np.log(np.abs(np.diff(GSC))), label='Trapezoidal')
# # plt.plot(BZgrid,GSD-mathematicapi, label='Monte Carlo')
# plt.plot(BZgrid,np.log(np.abs(np.diff(GSE))), label='Simpsons Rule')
#
#
# # plt.plot(JP, MFE0old, label='0 old')
# # plt.plot(JP, MFEpiold, label=r'$\pi$ old')
# plt.legend()
# plt.show()

# findPhaseMag100(0,0.05,1,0,0.05,1,h100,25,2,'test')
# JPsmol = np.linspace(-0.2, 1, 100)
# hsmol = np.linspace(0, 1, 100)
#
# phases = np.loadtxt('Nia_Phase_Diagrams/phase_111_kappa=2.txt')
# phases = np.flip(np.where((phases==1)|(phases==6)|(phases==10)|(phases==11)|(phases==15)|(phases==16), 16, phases), axis=0)
#
# phases = np.where(phases==15, 8, phases)
#
#
# graphMagPhase(JPsmol, hsmol, phases, 'test.pdf')


# n = 20
# h = 0.3
# BZres = 35
# JP = np.linspace(-0.05,0.04, n)
#
# GS0 = np.zeros(n)
# GSpi = np.zeros(n)
# GSpp00 = np.zeros(n)
# GSzzpp = np.zeros(n)
#
# MFE0 = np.zeros(n)
# MFEpi = np.zeros(n)
# MFEzzpp = np.zeros(n)
# MFEpp00 = np.zeros(n)
#
# for i in range(n):
#     A = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h111, flux=np.zeros(4))
#     B = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h111, flux=np.array([np.pi, np.pi, 0, 0]))
#     D = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h111, flux=np.array([0, 0, np.pi, np.pi]))
#     C = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h111, flux=np.array([np.pi, np.pi, np.pi, np.pi]))
#
#     A.solvemeanfield()
#     # B.solvemeanfield()
#     # D.solvemeanfield()
#     C.solvemeanfield()
#     # print(D.MFE(),D.GS())
#
#     # GS0[i] = A.GS()
#     # print(GS0[i])
#     GS0[i] = A.MFE()
#     # print(GS0[i])
#     # print(JP[i], A.minLams, A.lams, A.condensed, A.rhos)
#     # GSpp00[i] = B.MFE()
#     # GSzzpp[i] = D.MFE()
#     GSpi[i] = C.MFE()
#
#     #
#     print(JP[i], GS0[i], A.qmin, GSpp00[i], B.qmin, GSzzpp[i], D.qmin, GSpi[i], C.qmin)
#
# # plt.plot(JP, MFE0, label='0')
# # plt.plot(JP, MFEpp00, label=r'$\pi\pi 0 0$')
# # plt.plot(JP, MFEpi, label=r'$\pi$')
# # # plt.plot(JP, MFE0old, label='0 old')
# # # plt.plot(JP, MFEpiold, label=r'$\pi$ old')
# # plt.legend()
# # plt.show()
#
# plt.plot(JP, GS0, label='0')
# # plt.plot(JP, GSpp00, label=r'$ \pi\pi 00$')
# # plt.plot(JP, GSzzpp, label=r'$0 0\pi \pi $')
# plt.plot(JP, GSpi, label=r'$\pi$')
# # plt.plot(JP, MFE0old, label='0 old')
# # plt.plot(JP, MFEpiold, label=r'$\pi$ old')
# plt.legend()
# plt.show()
# flux = np.array([np.pi, np.pi, 0, 0])
# findPhaseMag(-0.5, 0.1, 25, 0, 0.3, 25, h110, 2, 2, flux, 'test')
# completeSpan(0, 1, 1, 0, 1, 1, h110, 30,2,np.zeros(4),'test')

# flux = np.ones(4)*np.pi
# # flux=np.zeros(4)
# # flux = np.array([np.pi, np.pi, 0, 0])
# JP = -0.04
# h= 0
# BZres = 30
# A = pycon.piFluxSolver(JP, JP, 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=flux)
# A.solvemeanfield()
# A.MFE()
# M = A.M_true(A.qmin)
# E, V = np.linalg.eigh(M)
# print(E, A.lams, E[0]+A.lams[0])
# B = np.where(A.qmin>0.5, A.qmin-1, A.qmin)
# print(A.qmin, contract('ij,jk->ik', A.qmin, BasisBZA), B, contract('ij,jk->ik', B, BasisBZA))

# A.graph(True)
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