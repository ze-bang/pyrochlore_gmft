import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import warnings
import pyrochlore_gmft as pycon
from variation_flux import *
from phase_diagram import *
import pyrochlore_exclusive_boson as pyeb
from observables import *

def ex_vs_gauge_gs_110(h, n, filename, solvemeanfield=False):
    Jpm = np.linspace(-0.05, 0.05, 30)

    GS0 = np.zeros(30)
    GSpi = np.zeros(30)
    GSpp00 = np.zeros(30)
    GS00pp = np.zeros(30)

    GS0h = np.zeros(30)
    GSpih = np.zeros(30)
    GSpp00h = np.zeros(30)
    GS00pph = np.zeros(30)


    for i in range(30):
        py0 = pyeb.piFluxSolver(-2*Jpm[i], -2*Jpm[i], 1, kappa=2, graphres=graphres, BZres=25, h=h, n=n, flux=np.zeros(4))
        pypi = pyeb.piFluxSolver(-2*Jpm[i], -2*Jpm[i], 1, kappa=2, graphres=graphres, BZres=25, h=h, n=n, flux=np.ones(4)*np.pi)
        pypp00 = pyeb.piFluxSolver(-2*Jpm[i], -2*Jpm[i], 1, kappa=2, graphres=graphres, BZres=25, h=h, n=n, flux=np.array([np.pi,np.pi,0,0]))
        py00pp = pyeb.piFluxSolver(-2*Jpm[i], -2*Jpm[i], 1, kappa=2, graphres=graphres, BZres=25, h=h, n=n, flux=np.array([0,0,np.pi,np.pi]))
        if solvemeanfield:
            py0.solvemeanfield()
            pypi.solvemeanfield()
            pypp00.solvemeanfield()
            py00pp.solvemeanfield()

        GS0[i] = py0.GS()
        GSpi[i] = pypi.GS()
        GSpp00[i] = pypp00.GS()
        GS00pp[i] = py00pp.GS()

        GS0h[i] = HanYan_GS(Jpm[i], 1, h, n, np.zeros(4))
        GSpih[i] = HanYan_GS(Jpm[i], 1, h, n, np.ones(4)*np.pi)
        GSpp00h[i] = HanYan_GS(Jpm[i], 1, h, n, np.array([np.pi,np.pi,0,0]))
        GS00pph[i] = HanYan_GS(Jpm[i], 1, h, n, np.array([0,0,np.pi,np.pi]))

    np.savetxt(filename+"_0_flux_ex.txt",GS0)
    np.savetxt(filename+"_0_flux_gauge.txt",GS0h)
    np.savetxt(filename+"_pi_flux_ex.txt",GSpi)
    np.savetxt(filename+"_pi_flux_gauge.txt",GSpih)
    np.savetxt(filename+"_pp00_flux_ex.txt",GSpp00)
    np.savetxt(filename+"_pp00_flux_gauge.txt",GSpp00h)
    np.savetxt(filename+"_00pp_flux_ex.txt",GS00pp)
    np.savetxt(filename+"_00pp_flux_ex.txt",GS00pph)


    plt.plot(Jpm, GS0-GS0, Jpm, GSpi-GS0, Jpm, GSpp00-GS0, Jpm, GS00pp-GS0)
    plt.legend([r'$0$', r'$\pi$', r'$\pi\pi 00$', r'$00\pi\pi$'])
    plt.savefig(filename+"_ex.pdf")
    plt.clf()
    plt.plot(Jpm, GS0h-GS0h, Jpm, GSpih-GS0h, Jpm, GSpp00h-GS0h, Jpm, GS00pph-GS0h)
    plt.legend([r'$0$', r'$\pi$', r'$\pi\pi 00$', r'$00\pi\pi$'])
    plt.savefig(filename+"_gauge.pdf")
    plt.clf()

    GS0tot = GS0+GS0h
    GSpitot = GSpi+GSpih
    GSpp00tot = GSpp00+GSpp00h
    GS00pptot = GS00pp+GS00pph

    plt.plot(Jpm, GS0tot-GS0tot, Jpm, GSpitot-GS0tot, Jpm, GSpp00tot-GS0tot, Jpm, GS00pptot-GS0tot)
    plt.legend([r'$0$', r'$\pi$', r'$\pi\pi 00$', r'$00\pi\pi$'])
    plt.savefig(filename+"_total.pdf")
    plt.clf()
def ex_vs_gauge_gs_111(h, n, filename, solvemeanfield=False):
    Jpm = np.linspace(-0.05, 0.05, 30)

    GS0 = np.zeros(30)
    GSpi = np.zeros(30)
    GS0h = np.zeros(30)
    GSpih = np.zeros(30)

    for i in range(30):
        py0 = pyeb.piFluxSolver(-2*Jpm[i], -2*Jpm[i], 1, kappa=2, graphres=graphres, BZres=25, h=h, n=n, flux=np.zeros(4))
        pypi = pyeb.piFluxSolver(-2*Jpm[i], -2*Jpm[i], 1, kappa=2, graphres=graphres, BZres=25, h=h, n=n, flux=np.ones(4)*np.pi)
        if solvemeanfield:
            py0.solvemeanfield()
            pypi.solvemeanfield()

        GS0[i] = py0.GS()
        GSpi[i] = pypi.GS()

        GS0h[i] = HanYan_GS(Jpm[i], 1, h, n, np.zeros(4))
        GSpih[i] = HanYan_GS(Jpm[i], 1, h, n, np.ones(4)*np.pi)


    np.savetxt(filename+"_0_flux_ex.txt",GS0)
    np.savetxt(filename+"_0_flux_gauge.txt",GS0h)
    np.savetxt(filename+"_pi_flux_ex.txt",GSpi)
    np.savetxt(filename+"_pi_flux_gauge.txt",GSpih)



    plt.plot(Jpm, GS0-GS0, Jpm, GSpi-GS0)
    plt.legend([r'$0$', r'$\pi$'])
    plt.savefig(filename+"_ex.pdf")
    plt.clf()
    plt.plot(Jpm, GS0h-GS0h, Jpm, GSpih-GS0h)
    plt.legend([r'$0$', r'$\pi$'])
    plt.savefig(filename+"_gauge.pdf")
    plt.clf()

    GS0tot = GS0+GS0h
    GSpitot = GSpi+GSpih

    plt.plot(Jpm, GS0tot-GS0tot, Jpm, GSpitot-GS0tot)
    plt.legend([r'$0$', r'$\pi$'])
    plt.savefig(filename+"_total.pdf")
    plt.clf()
def ex_vs_gauge_gs_001(h, n, filename, solvemeanfield=False):
    Jpm = np.linspace(-0.05, 0.05, 30)

    GS0 = np.zeros(30)
    GSpi = np.zeros(30)
    GSpp00 = np.zeros(30)
    GS00pp = np.zeros(30)

    GS0h = np.zeros(30)
    GSpih = np.zeros(30)
    GSpp00h = np.zeros(30)
    GS00pph = np.zeros(30)


    for i in range(30):
        py0 = pyeb.piFluxSolver(-2*Jpm[i], -2*Jpm[i], 1, kappa=2, graphres=graphres, BZres=25, h=h, n=n, flux=np.zeros(4))
        pypi = pyeb.piFluxSolver(-2*Jpm[i], -2*Jpm[i], 1, kappa=2, graphres=graphres, BZres=25, h=h, n=n, flux=np.ones(4)*np.pi)
        pypp00 = pyeb.piFluxSolver(-2*Jpm[i], -2*Jpm[i], 1, kappa=2, graphres=graphres, BZres=25, h=h, n=n, flux=np.array([0,np.pi,np.pi,0]))
        py00pp = pyeb.piFluxSolver(-2*Jpm[i], -2*Jpm[i], 1, kappa=2, graphres=graphres, BZres=25, h=h, n=n, flux=np.array([np.pi,0,0,np.pi]))
        if solvemeanfield:
            py0.solvemeanfield()
            pypi.solvemeanfield()
            pypp00.solvemeanfield()
            py00pp.solvemeanfield()

        GS0[i] = py0.GS()
        GSpi[i] = pypi.GS()
        GSpp00[i] = pypp00.GS()
        GS00pp[i] = py00pp.GS()

        GS0h[i] = HanYan_GS(Jpm[i], 1, h, n, np.zeros(4))
        GSpih[i] = HanYan_GS(Jpm[i], 1, h, n, np.ones(4)*np.pi)
        GSpp00h[i] = HanYan_GS(Jpm[i], 1, h, n, np.array([0,np.pi,np.pi,0]))
        GS00pph[i] = HanYan_GS(Jpm[i], 1, h, n, np.array([np.pi,0,0,np.pi]))

    np.savetxt(filename+"_0_flux_ex.txt",GS0)
    np.savetxt(filename+"_0_flux_gauge.txt",GS0h)
    np.savetxt(filename+"_pi_flux_ex.txt",GSpi)
    np.savetxt(filename+"_pi_flux_gauge.txt",GSpih)
    np.savetxt(filename+"_0pp0_flux_ex.txt",GSpp00)
    np.savetxt(filename+"_0pp0_flux_gauge.txt",GSpp00h)
    np.savetxt(filename+"_p00p_flux_ex.txt",GS00pp)
    np.savetxt(filename+"_p00p_flux_gauge.txt",GS00pph)


    plt.plot(Jpm, GS0-GS0, Jpm, GSpi-GS0, Jpm, GSpp00-GS0, Jpm, GS00pp-GS0)
    plt.legend([r'$0$', r'$\pi$', r'$0\pi\pi 0$', r'$\pi 00\pi$'])
    plt.savefig(filename+"_ex.pdf")
    plt.clf()
    plt.plot(Jpm, GS0h-GS0h, Jpm, GSpih-GS0h, Jpm, GSpp00h-GS0h, Jpm, GS00pph-GS0h)
    plt.legend([r'$0$', r'$\pi$', r'$0\pi\pi 0$', r'$\pi 00\pi$'])
    plt.savefig(filename+"_gauge.pdf")
    plt.clf()

    GS0tot = GS0+GS0h
    GSpitot = GSpi+GSpih
    GSpp00tot = GSpp00+GSpp00h
    GS00pptot = GS00pp+GS00pph

    plt.plot(Jpm, GS0tot-GS0tot, Jpm, GSpitot-GS0tot, Jpm, GSpp00tot-GS0tot, Jpm, GS00pptot-GS0tot)
    plt.legend([r'$0$', r'$\pi$', r'$0\pi\pi 0$', r'$\pi 00\pi$'])
    plt.savefig(filename+"_total.pdf")
    plt.clf()
def regraph(dir, hhl):
    H = np.linspace(-2.5, 2.5, 100)
    L = np.linspace(-2.5, 2.5, 100)
    A, B = np.meshgrid(H, L)

    for dirs in os.listdir(dir):
        foldname = dir+dirs
        for files in os.listdir(foldname):
            if files.endswith('.txt'):
                temp = foldname+'/'+files
                d = np.loadtxt(temp)
                d = d/np.max(d)
                vmin = 0
                vmax = 1
                if files.endswith('NSF.txt'):
                    d = d/2
                    vmax = 0.5
                if files.endswith('global.txt'):
                    vmin = 0.6

                if hhl == "hhl":
                    SSSFGraphHHL(A, B, d, temp[:-4], 2.5, 2.5, vmin, vmax)
                elif hhl == "hk0":
                    SSSFGraphHK0(A, B, d, temp[:-4], 2.5, 2.5, vmin, vmax)
                else:
                    SSSFGraphHKK(A, B, d, temp[:-4], 2.5, 2.5, vmin, vmax)

def regraphDSSF(dir):
    for dirs in os.listdir(dir):
        foldname = dir+dirs
        for files in os.listdir(foldname):
            if files.endswith('.txt'):
                temp = foldname+'/'+files
                d = np.loadtxt(temp)
                d = d/np.max(d)
                kline = np.concatenate((graphGammaX, graphXW, graphWK, graphKGamma, graphGammaL, graphLU, graphUW))
                e = np.linspace(0,1,len(d))
                X, Y = np.meshgrid(kline, e)
                DSSFgraph(X, Y, d.T, temp[:-4])

# Jpm = 0.02
# h =0.3
# n=h110
# # a = pycon.piFluxSolver(-2*Jpm, 1, -2*Jpm, h=h,n=n,flux=np.zeros(4))
# # a.solvemeanfield()
# # a.graph(False)
# # plt.savefig("h110_Jpm="+str(Jpm)+"h="+str(h)+".pdf")
# # plt.clf()
# #
# # Jpm = -0.05
# # h =0.3
# # a = pycon.piFluxSolver(-2*Jpm, 1, -2*Jpm, h=h,n=n,flux=np.array([0,0,np.pi,np.pi]))
# # a.solvemeanfield()
# # a.graph(False)
# # plt.savefig("h110_Jpm="+str(Jpm)+"h="+str(h)+".pdf")
# # plt.clf()
# dir = "cedar_DSSF/"
# regraphDSSF(dir)



Jpm = -0.289
h = 0.21
n=h110
a = pycon.piFluxSolver(-2*Jpm, 1, -2*Jpm, h=h,n=n,flux=np.array([np.pi,np.pi,np.pi,np.pi]))
a.solvemeanfield()
print(a.magnetization(),a.lams-a.minLams,a.condensed)

#
# Jpm = -0.2
# a = pycon.piFluxSolver(-2*Jpm, -2*Jpm, 1)
# a.solvemeanfield()
# C1 = a.green_pi(a.pts)
# A1, B2 = SpmSpp(a.pts, a.pts,np.array([0,0,0]), a, a.lams)
# Szz = (np.real(A1) + np.real(B2)) / 2
# qreal = np.array([0, 0, 0])
# Sglobalzz2 = contract('ijk,jk,i->jk', Szz, g(qreal), a.weights)
# Szz2 = contract('ijk,i->jk', Szz, a.weights)
# print(g(qreal),Szz2, Sglobalzz2)

# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.03_pi/h_110/"
# regraph(dir, "hhl")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.03_0/h_110/"
# regraph(dir, "hhl")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.03_00pp/h_110/"
# regraph(dir, "hhl")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=0.02_0/h_110/"
# regraph(dir, "hhl")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=0.02_pi/h_110/"
# regraph(dir, "hhl")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=0.02_00pp/h_110/"
# regraph(dir, "hhl")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.289_pi/h_110/"
# regraph(dir, "hhl")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.289_0/h_110/"
# regraph(dir, "hhl")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.289_00pp/h_110/"
# regraph(dir, "hhl")
#
#
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.03_pi/h_111/"
# regraph(dir, "hkk")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.03_0/h_111/"
# regraph(dir, "hkk")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=0.02_0/h_111/"
# regraph(dir, "hkk")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=0.02_pi/h_111/"
# regraph(dir, "hkk")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.289_pi/h_111/"
# regraph(dir, "hkk")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.289_0/h_111/"
# regraph(dir, "hkk")
#
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.03_pi/h_001/"
# regraph(dir, "hk0")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.03_0/h_001/"
# regraph(dir, "hk0")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=0.02_0/h_001/"
# regraph(dir, "hk0")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=0.02_pi/h_001/"
# regraph(dir, "hk0")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.289_pi/h_001/"
# regraph(dir, "hk0")
# dir = "Nia_SSSF_DSSF/SSSF/octupolar/Jpm=-0.289_0/h_001/"
# regraph(dir, "hk0")
#
#
