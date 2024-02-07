import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import pyrochlore_conclusive as pycon
from variation_flux import *


# getminflux110("Niagara_data_3/h110_flux_plane_zero_0.2")
# getminflux110("Niagara_data_3/h110_flux_plane_pi_0.2")
# getminflux110("Niagara_data_3/h110_flux_plane_mid_0.2")
#
# getminflux111("Niagara_data_3/h111_flux_plane_zero_0.2")
# getminflux111("Niagara_data_3/h111_flux_plane_pi_0.2")
# getminflux111("Niagara_data_3/h111_flux_plane_mid_0.2")
ppp0f = generateflux110(0, np.pi, 0, 0)
pp00f = generateflux110(np.pi, 0, 1, 0)
# JP = 0.045
# h = 0
# BZres = 20
#
# z0 = pycon.piFluxSolver(-2 * JP, -2 * JP, 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=np.zeros(4))
# z0.solvemeanfield()
# print(z0.condensed, z0.lams, z0.minLams, z0.qmin, z0.GS(), z0.rhos, z0.delta)
# z0.graph(True)
#
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


n = 20
h = 0
BZres = 20
JP = np.linspace(-0.01,0.01, n)

GS0 = np.zeros(n)
GSpi = np.zeros(n)
GSppp0 = np.zeros(n)
GSpp00 = np.zeros(n)

for i in range(n):
    A = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=np.zeros(4))
    B = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=ppp0f)
    D = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=pp00f)
    C = pycon.piFluxSolver(-2*JP[i], -2*JP[i], 1, kappa=2, graphres=graphres, BZres=BZres, h=h, n=h110, flux=np.array([np.pi, np.pi, np.pi, np.pi]))

    A.solvemeanfield()
    B.solvemeanfield()
    D.solvemeanfield()
    C.solvemeanfield()

    GS0[i] = A.GS()
    GSppp0[i] = B.GS()
    GSpp00[i] = D.GS()
    GSpi[i] = C.GS()


    print(JP[i], GS0[i], A.qmin, GSppp0[i], B.qmin, GSpp00[i], D.qmin, GSpi[i], C.qmin)

plt.plot(JP, GS0, label='0')
plt.plot(JP, GSpp00, label=r'$\pi\pi 0 0$')
plt.plot(JP, GSppp0, label=r'$\pi\pi \pi 0$')
plt.plot(JP, GSpi, label=r'$\pi$')
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