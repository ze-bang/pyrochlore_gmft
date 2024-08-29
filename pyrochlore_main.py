import os
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['cm']
# import matplotlib.pyplot as plt
import numpy as np

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import warnings
import pyrochlore_gmft as pycon
from variation_flux import *
from phase_diagram import *
import pyrochlore_exclusive_boson as pyeb
from observables import *
import netCDF4 as nc

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
    if hhl == "hnhkk2k":
        Hr = 3
        Lr = 3
    else:
        Hr = 2.5
        Lr = 2.5

    for dirs in os.listdir(dir):
        foldname = dir+'/'+dirs
        if dirs == "h=0.0":
            dzero = np.loadtxt(foldname+'/Szzglobal.txt')
            dzero = dzero/np.max(dzero)
        else:
            d = np.loadtxt(foldname+'/Szzglobal.txt')
            d = d/np.max(d) - dzero
            SSSFGraphHnHL(d, foldname+'/Szz_0_subtracted', Hr, Lr)

def regraphSSSFhelper(d, foldname, hhl):
    if hhl == "110":
        SSSFGraphHnHL(d, foldname, 2.5, 2.5)
    elif hhl == "111":
        SSSFGraphHH2K(d, foldname, 3, 3)
    else:
        SSSFGraphHK0(d, foldname, 2.5, 2.5)
from matplotlib.ticker import FuncFormatter

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)
def SSSFgenhelper(d, hhl, ax, fig):

    if hhl == "110":
        c = ax.imshow(d, interpolation="lanczos", origin='lower', extent=[-2.5, 2.5, -2.5, 2.5], aspect='auto')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
    elif hhl == "111":
        c = ax.imshow(d, interpolation="lanczos", origin='lower', extent=[-3, 3, -3, 3], aspect='auto')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
    else:
        c = ax.imshow(d, interpolation="lanczos", origin='lower', extent=[-2.5, 2.5, -2.5, 2.5], aspect='auto')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
    cb = fig.colorbar(c, ax=ax)

def regraphSSSF(dir):
    for jpm in os.listdir(dir):
        for hdir in os.listdir(dir+'/'+jpm):
            dirstring = hdir[2:]
            for dirs in os.listdir(dir+'/'+jpm+'/'+hdir):
                foldname = dir+'/'+jpm+'/'+hdir+'/'+dirs
                d = np.loadtxt(foldname+'/Szzglobal.txt')
                dmax = np.max(d)
                regraphSSSFhelper(d/dmax, foldname+'/Szzglobal', dirstring)
                for g in os.listdir(foldname+'/Szzglobal'):
                    if g.endswith('.txt'):
                        subname = foldname+'/Szzglobal/' + g
                        d = np.loadtxt(subname)
                        regraphSSSFhelper(d/dmax, subname[:-4], dirstring)

                d = np.loadtxt(foldname+'/Szz.txt')
                dmax = np.max(d)
                regraphSSSFhelper(d/dmax, foldname+'/Szz', dirstring)
                for g in os.listdir(foldname+'/Szz'):
                    if g.endswith('.txt'):
                        subname = foldname+'/Szz/' + g
                        d = np.loadtxt(subname)
                        regraphSSSFhelper(d/dmax, subname[:-4], dirstring)

                d = np.loadtxt(foldname+'/Sxxglobal.txt')
                dmax = np.max(d)
                regraphSSSFhelper(d/dmax, foldname+'/Sxxglobal', dirstring)
                for g in os.listdir(foldname+'/Sxxglobal'):
                    if g.endswith('.txt'):
                        subname = foldname+'/Sxxglobal/' + g
                        d = np.loadtxt(subname)
                        regraphSSSFhelper(d/dmax, subname[:-4], dirstring)

                d = np.loadtxt(foldname+'/Sxx.txt')
                dmax = np.max(d)
                regraphSSSFhelper(d/dmax, foldname+'/Sxx', dirstring)
                for g in os.listdir(foldname+'/Sxx'):
                    if g.endswith('.txt'):
                        subname = foldname+'/Sxx/' + g
                        d = np.loadtxt(subname)
                        regraphSSSFhelper(d/dmax, subname[:-4], dirstring)



def regraphDSSF(dir):
    for dirs in os.listdir(dir):
        foldname = dir+dirs
        for files in os.listdir(foldname):
            if files.endswith('.txt'):
                fig, axs = plt.subplots()
                temp = foldname+'/'+files
                d = np.loadtxt(temp)
                d = d/np.max(d)
                a = pycon.piFluxSolver(-0.04, -0.04, 1, flux=np.zeros(4), h=0, n=h110)
                a.solvemeanfield()
                emin, emax = a.graph_loweredge(False, axs), a.graph_upperedge(False, axs)
                c = axs.imshow(d.T / np.max(d), interpolation="lanczos", origin='lower',
                                     extent=[0, gGamma3, emin, emax], aspect='auto', cmap='gnuplot2')
                fig.colorbar(c, ax=axs)

def regraphPhase(dir):
    for files in os.listdir(dir):
        if files.endswith('.txt'):
            temp = dir+'/'+files
            d = np.loadtxt(temp)
            JP = np.linspace(-0.5,0.1,300)
            if files[6:9]=="111":
                h = np.linspace(0, 0.7, 200)
            elif files[6:9] == "100":
                h = np.linspace(0, 0.5, 200)
            else:
                h = np.linspace(0, 2.2, 200)


            plt.imshow(d.T, interpolation="nearest", origin='lower', extent=[-0.5, 0.1, np.min(h), np.max(h)], aspect='auto')
            plt.colorbar()
            plt.xlabel(r'$J_\pm/J_{yy}$')
            plt.ylabel(r'$h/J_{yy}$')
            plt.savefig(temp[:-4] + '.pdf')
            plt.clf()

def DSSFparse(emin, emax, d):
    d = d.T
    dn = (emax-emin)/len(d)
    le = d.shape[1]
    # print(dn, le)
    zfill = np.zeros((int(emin/dn),le))
    d = np.concatenate((zfill,d))
    return d.T
def DSSFgraphGen(h0, hmid, mid, hpi, n, filename):
    mpl.rcParams.update({'font.size': 22})
    plt.margins(x=0.04,y=0.04)
    if not np.isnan(hpi):
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(22, 8), layout="constrained", sharex=True)
        axs[0,0].text(.01, .99, r"$(\mathrm{a})\; 0$-$\mathrm{flux}$", ha='left', va='top', transform=axs[0,0].transAxes, zorder=10)
        axs[0,1].text(.01, .99, r"$(\mathrm{b})\; (0,\pi,\pi,0)$", ha='left', va='top', transform=axs[0,1].transAxes, zorder=10)
        axs[0,2].text(.01, .99, r"$(\mathrm{c})\; \pi$-$\mathrm{flux}$", ha='left', va='top', transform=axs[0,2].transAxes, zorder=10)
        axs[1,0].text(.01, .95, r"$(\mathrm{d})$", ha='left', va='top', transform=axs[1,0].transAxes, color='w', zorder=10)
        axs[1,1].text(.01, .95, r"$(\mathrm{e})$", ha='left', va='top', transform=axs[1,1].transAxes, color='w', zorder=10)
        axs[1,2].text(.01, .95, r"$(\mathrm{f})$", ha='left', va='top', transform=axs[1,2].transAxes, color='w', zorder=10)
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8), layout="constrained", sharex=True)
        axs[0, 0].text(.01, .99, r"$(\mathrm{a})\; 0$-$\mathrm{flux}$", ha='left', va='top', transform=axs[0, 0].transAxes,
                       zorder=10)
        if (mid==np.array([0,0,np.pi,np.pi])).all():
            axs[0, 1].text(.01, .99, r"$(\mathrm{b})\; (0,\pi,\pi,0)$", ha='left', va='top', transform=axs[0, 1].transAxes,
                           zorder=10)
        else:
            axs[0, 1].text(.01, .99, r"$(\mathrm{a})\; \pi$-$\mathrm{flux}$", ha='left', va='top', transform=axs[0, 1].transAxes,
                           zorder=10)
        axs[1, 0].text(.01, .95, r"$(\mathrm{c})$", ha='left', va='top', transform=axs[1, 0].transAxes, color='w',
                       zorder=10)
        axs[1, 1].text(.01, .95, r"$(\mathrm{d})$", ha='left', va='top', transform=axs[1, 1].transAxes, color='w',
                       zorder=10)

    axs[0,0].set_ylabel(r'$\omega/J_{yy}$')
    axs[1,0].set_ylabel(r'$\omega/J_{yy}$')

    dirString = ""
    if (n==h110).all():
        dirString = "110"
    elif (n==h111).all():
        dirString = "111"
    else:
        dirString = "001"

    fluxString = ""
    if (mid==np.zeros(4)).all():
        fluxString = "0"
    elif (mid==np.ones(4)*np.pi).all():
        fluxString = "pi"
    else:
        fluxString = "00pp"

    # fig.supylabel(r'$\omega/J_{yy}$')

    a = pycon.piFluxSolver(-0.06,-0.06,1,flux=np.zeros(4),h=h0,n=n, simplified=True)
    a.solvemeanfield()
    a.graph(axs[0,0])

    emin, emax = a.graph_loweredge(False, axs[1,0]), a.graph_upperedge(False, axs[1,0])
    d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=0.03_0/h_"+dirString+"/h="+str(h0)+"/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    d = DSSFparse(emin, emax, d)
    c = axs[1,0].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, 0, emax], aspect='auto', cmap='gnuplot2')
    fig.colorbar(c, ax=axs[1,0])

    a = pycon.piFluxSolver(0.06,0.06,1,flux=mid,h=hmid,n=n, simplified=True)
    a.solvemeanfield()
    a.graph(axs[0,1])

    emin, emax = a.graph_loweredge(False, axs[1,1]), a.graph_upperedge(False, axs[1,1])
    d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.03_"+fluxString+"/h_"+dirString+"/h="+str(hmid)+"/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    d = DSSFparse(emin, emax, d)
    c = axs[1,1].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, 0, emax], aspect='auto', cmap='gnuplot2')
    fig.colorbar(c, ax=axs[1,1])

    if not np.isnan(hpi):
        a = pycon.piFluxSolver(0.289*2,0.289*2,1,flux=np.ones(4)*np.pi,h=hpi,n=n, simplified=True)
        a.solvemeanfield()
        a.graph(axs[0,2])

        emin, emax = a.graph_loweredge(False, axs[1,2]), a.graph_upperedge(False, axs[1,2])
        d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_"+dirString+"/h="+str(hpi)+"/Szzglobal.txt")
        emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
        c = axs[1,2].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax], aspect='auto', cmap='gnuplot2')
        fig.colorbar(c, ax=axs[1,2])


    plt.savefig(filename)
    plt.clf()

def DSSFgraphGen_0(n, filename):
    mpl.rcParams.update({'font.size': 25})
    plt.margins(x=0.04,y=0.04)

    fig, axs = plt.subplots(nrows=2,ncols=2, figsize=(16, 8),layout="constrained", sharex=True)

    axs[0,0].text(.01, .99, r"$(\mathrm{a})$", ha='left', va='top', transform=axs[0,0].transAxes, zorder=10)
    axs[0,1].text(.01, .99, r"$(\mathrm{b})$", ha='left', va='top', transform=axs[0,1].transAxes, zorder=10)
    axs[1,0].text(.01, .95, r"$(\mathrm{c})$", ha='left', va='top', transform=axs[1,0].transAxes, color='w', zorder=10)
    axs[1,1].text(.01, .95, r"$(\mathrm{d})$", ha='left', va='top', transform=axs[1,1].transAxes, color='w', zorder=10)
    axs[0,0].set_ylabel(r'$\omega/J_{yy}$')
    axs[1,0].set_ylabel(r'$\omega/J_{yy}$')
    dirString = ""
    if (n==h110).all():
        dirString = "110"
    elif (n==h111).all():
        dirString = "111"
    else:
        dirString = "001"

    a = pycon.piFluxSolver(-0.06,-0.06,1,flux=np.zeros(4),h=0,n=n, simplified=True)
    a.solvemeanfield()
    a.graph(axs[0,0])

    emin, emax = a.graph_loweredge(False, axs[1,0]), a.graph_upperedge(False, axs[1,0])
    d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=0.03_0/h_"+dirString+"/h=0.0/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    d = DSSFparse(emin, emax, d)
    c = axs[1,0].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, 0, emax], aspect='auto', cmap='gnuplot2')
    fig.colorbar(c, ax=axs[1,0])


    a = pycon.piFluxSolver(0.289*2,0.289*2,1,flux=np.ones(4)*np.pi,h=0,n=n)
    a.solvemeanfield()
    a.graph(axs[0,1])

    emin, emax = a.graph_loweredge(False, axs[1,1]), a.graph_upperedge(False, axs[1,1])
    d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_"+dirString+"/h=0.0/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    d = DSSFparse(emin, emax, d)
    c = axs[1,1].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, 0, emax], aspect='auto', cmap='gnuplot2')
    fig.colorbar(c, ax=axs[1,1])

    plt.savefig(filename)

def SSSFgraphGen(h0, hmid, fluxmid, hpi, n, tograph, filename, colors=np.array(['w','w','w','w','w','w','w','w','w','w'])):
    mpl.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(nrows=3,ncols=3, figsize=(16, 14), layout="tight", sharex=True, sharey=True)

    if len(hpi) == 2:
        axs[-1, -1].axis('off')

    axs[0,0].set_xlabel(r'$J_\pm/J_{yy}=0.03$', labelpad=10)
    axs[0,0].xaxis.set_label_position('top')
    axs[0,1].set_xlabel(r'$J_\pm/J_{yy}=-0.03$', labelpad=10)
    axs[0,1].xaxis.set_label_position('top')
    axs[0,2].set_xlabel(r'$J_\pm/J_{yy}=-0.3$', labelpad=10)
    axs[0,2].xaxis.set_label_position('top')


    htoprint = np.around(2*h0,decimals=1)/2
    fig.text(1, 0.85, r'$h/J_{yy}=' + str(htoprint[0]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.52, r'$h/J_{yy}=' + str(htoprint[1]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.2, r'$h/J_{yy}=' + str(htoprint[2]) + '$', ha='right', va='center', rotation=270)

    dirString = ""
    fluxString = ""
    if (n==h110).all():
        dirString = "110"
        plthhlfBZ(axs[0,0])

        axs[0, 0].set_ylabel(r'$(0,0,l)$')
        axs[1, 0].set_ylabel(r'$(0,0,l)$')
        axs[2, 0].set_ylabel(r'$(0,0,l)$')
        axs[2, 0].set_xlabel(r'$(h,-h,0)$')
        axs[2, 1].set_xlabel(r'$(h,-h,0)$')
        axs[1, 2].set_xlabel(r'$(h,-h,0)$')
        fluxString="(0,\pi,\pi,0)"
    elif (n==h111).all():
        dirString = "111"
        plthh2kfBZ(axs[0,0])

        axs[0, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[1, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[2, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[2, 0].set_xlabel(r'$(h,-h,0)$')
        axs[2, 1].set_xlabel(r'$(h,-h,0)$')
        axs[2, 2].set_xlabel(r'$(h,-h,0)$')
        fluxString="\pi\; flux"

    else:
        dirString = "001"
        plthk0fBZ(axs[0,0])

        axs[0, 0].set_ylabel(r'$(0,k,0)$')
        axs[1, 0].set_ylabel(r'$(0,k,0)$')
        axs[2, 0].set_ylabel(r'$(0,k,0)$')
        axs[2, 0].set_xlabel(r'$(h,0,0)$')
        axs[2, 1].set_xlabel(r'$(h,0,0)$')
        axs[1, 2].set_xlabel(r'$(h,0,0)$')
        fluxString="0\; flux"


    for i in range(len(h0)):
        d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=0.03_0/h_"+dirString+"/h="+str(h0[i])+"/"+tograph)
        if (np.mean(d[-10:, 0:30]) - np.min(d)) / (np.max(d) - np.min(d)) > 0.5:
            color = 'black'
        else:
            color = 'white'
        axs[i, 0].text(.01, .99, r"$"+chr(97+3*i)+")\; 0\; flux$", ha='left', va='top', transform=axs[i, 0].transAxes, color=color)
        SSSFgenhelper(d/np.max(d), dirString, axs[i,0], fig)

    for i in range(len(hmid)):
        d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.03_"+ fluxmid[i] +"/h_"+dirString+"/h="+str(hmid[i])+"/"+tograph)
        if (np.mean(d[-10:, 0:30]) - np.min(d)) / (np.max(d) - np.min(d)) > 0.5:
            color = 'black'
        else:
            color = 'white'
        if i < 2:
            axs[i, 1].text(.01, .99, r"$" + chr(98 + 3*i) + ")\; \pi\; flux$", ha='left', va='top', transform=axs[i, 1].transAxes,
                           color=color)
        else:
            axs[i, 1].text(.01, .99, r"$" + chr(98 + 3*i) + ")\;"+ fluxString +"$", ha='left', va='top', transform=axs[i, 1].transAxes,
                           color=color)
        SSSFgenhelper(d/np.max(d), dirString, axs[i,1], fig)

    for i in range(len(hpi)):
        d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_"+dirString+"/h="+str(hpi[i])+"/"+tograph)
        if (np.mean(d[-10:, 0:30]) - np.min(d)) / (np.max(d) - np.min(d)) > 0.5:
            color = 'black'
        else:
            color = 'white'
        axs[i, 2].text(.01, .99, r"$" + chr(99 + 3*i) + ")\; \pi\; flux$", ha='left', va='top', transform=axs[i, 2].transAxes,
                       color=color)
        SSSFgenhelper(d/np.max(d), dirString, axs[i,2], fig)

    plt.savefig(filename)
    plt.clf()

def sublatticeSSSFgraphGen(n, tograph, h, JP, flux, filename):
    mpl.rcParams.update({'font.size': 30})
    fig, axs = plt.subplots(nrows=len(h)*len(JP),ncols=len(tograph), figsize=(28, 35), layout="tight", sharex=True, sharey=True)
    dirString = ""
    if (n==h110).all():
        dirString = "110"
        plthhlfBZ(axs[0,0])
        axs[0, 0].set_ylabel(r'$(0,0,l)$')
        axs[1, 0].set_ylabel(r'$(0,0,l)$')
        axs[2, 0].set_ylabel(r'$(0,0,l)$')
        axs[3, 0].set_ylabel(r'$(0,0,l)$')
        axs[4, 0].set_ylabel(r'$(0,0,l)$')
        axs[5, 0].set_ylabel(r'$(0,0,l)$')
        axs[5, 0].set_xlabel(r'$(h,-h,0)$')
        axs[5, 1].set_xlabel(r'$(h,-h,0)$')
        axs[5, 2].set_xlabel(r'$(h,-h,0)$')
        axs[5, 3].set_xlabel(r'$(h,-h,0)$')

    elif (n==h111).all():
        dirString = "111"
        plthh2kfBZ(axs[0,0])

        axs[0, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[1, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[2, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[3, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[4, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[5, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[5, 0].set_xlabel(r'$(h,-h,0)$')
        axs[5, 1].set_xlabel(r'$(h,-h,0)$')
        axs[5, 2].set_xlabel(r'$(h,-h,0)$')
        axs[5, 3].set_xlabel(r'$(h,-h,0)$')
    else:
        dirString = "001"
        plthk0fBZ(axs[0,0])

        axs[0, 0].set_ylabel(r'$(0,k,0)$')
        axs[1, 0].set_ylabel(r'$(0,k,0)$')
        axs[2, 0].set_ylabel(r'$(0,k,0)$')
        axs[3, 0].set_ylabel(r'$(0,k,0)$')
        axs[4, 0].set_ylabel(r'$(0,k,0)$')
        axs[5, 0].set_ylabel(r'$(0,k,0)$')
        axs[5, 0].set_xlabel(r'$(h,0,0)$')
        axs[5, 1].set_xlabel(r'$(h,0,0)$')
        axs[5, 2].set_xlabel(r'$(h,0,0)$')
        axs[5, 3].set_xlabel(r'$(h,0,0)$')
    # htoprint = h
    htoprint = np.around(2*h,decimals=1)/2

    fig.text(1, 0.91, r'$h/J_{yy}=' + str(htoprint[0]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.74, r'$h/J_{yy}=' + str(htoprint[1]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.58, r'$h/J_{yy}=' + str(htoprint[2]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.42, r'$h/J_{yy}=' + str(htoprint[0]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.26, r'$h/J_{yy}=' + str(htoprint[1]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.1, r'$h/J_{yy}=' + str(htoprint[2]) + '$', ha='right', va='center', rotation=270)
    for i in range(len(tograph)):
        globalS = ""
        if "global" in tograph[i]:
            globalS = ''
        else:
            globalS = '\mathrm{LF},'
        temp = tograph[i].split("/")

        if (n==h111).all():
            axs[0, i].set_xlabel(
                r'$\mathcal{S}^{zz}_{' + globalS + temp[-1][0:-4] + '}\quad J_\pm=' + str(JP[0]) + '$ ',
                labelpad=10)
            axs[0, i].xaxis.set_label_position('top')
            axs[len(h), i].set_xlabel(
                r'$\mathcal{S}^{zz}_{' + globalS + temp[-1][0:-4] + '}\quad J_\pm=' + str(JP[1]) + '$',
                labelpad=10)
            axs[len(h), i].xaxis.set_label_position('top')
        else:
            axs[0, i].set_xlabel(r'$\mathcal{S}^{zz}_{'+globalS+temp[-1][0:-4]+'}\quad J_\pm='+str(JP[0])+'$ ', labelpad=10)
            axs[0, i].xaxis.set_label_position('top')
            axs[len(h), i].set_xlabel(r'$\mathcal{S}^{zz}_{'+globalS+temp[-1][0:-4]+'}\quad J_\pm='+str(JP[1])+'$', labelpad=10)
            axs[len(h), i].xaxis.set_label_position('top')

    for i in range(len(h)):
        for j in range(len(JP)):
            fluxString = flux[j,i]
            for k in range(len(tograph)):
                if "global" in tograph[k]:
                    dmax = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm="+str(JP[j])+"_" + fluxString + "/h_" + dirString + "/h=" + str(
                    h[i]) + "/Szzglobal.txt" )
                else:
                    dmax = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm="+str(JP[j])+"_" + fluxString + "/h_" + dirString + "/h=" + str(
                    h[i]) + "/Szz.txt")
                dmax = np.max(dmax)
                d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm="+str(JP[j])+"_" + fluxString + "/h_" + dirString + "/h=" + str(
                    h[i]) + "/" + tograph[k])
                SSSFgenhelper(d /dmax, dirString, axs[i+j*len(h),k], fig)
                if (np.mean(d[-10:,0:20])-np.min(d))/(np.max(d)-np.min(d))>0.5:
                    color = 'black'
                else:
                    color = 'white'
                axs[i+j*len(h),k].text(.01, .99, r"$(\mathrm{"+str(j+1)+chr(97+i*len(tograph)+k)+"})$", ha='left', va='top', transform=axs[i+j*len(h),k].transAxes, color=color)
    plt.savefig(filename)
    plt.clf()

def sublatticeSSSFgraphGen_select(n, tograph, h, JP, flux, fluxStringS, filename, s=0, tol=False):
    mpl.rcParams.update({'font.size': 35})
    yes = h.shape[1]==3
    yesJP = len(JP)==3
    if yes:
        length=20
        # fig, axs = plt.subplots(nrows=len(h),ncols=len(tograph), figsize=(16, 20), layout="tight", sharex=True, sharey=True)
    else:
        length=14

    if yesJP:
        width=24
    else:
        width=16

    fig, axs = plt.subplots(nrows=h.shape[1],ncols=len(tograph), figsize=(width, length), layout="tight", sharex=True, sharey=True)

    dirString = ""
    if (n==h110).all():
        dirString = "110"
        if s == 0:
            plthhlfBZ(axs[0,0])
        else:
            plt1dhhlfBZ(axs[0,0])
        axs[0, 0].set_ylabel(r'$(0,0,l)$')
        axs[-1, 0].set_ylabel(r'$(0,0,l)$')
        axs[-1, 0].set_xlabel(r'$(h,-h,0)$')
        axs[-1, 1].set_xlabel(r'$(h,-h,0)$')
        if yes:
            axs[1,0].set_ylabel(r'$(0,0,l)$')
        if yesJP:
            axs[1,-1].set_xlabel(r'$(h,-h,0)$')
    elif (n==h111).all():
        dirString = "111"
        plthh2kfBZ(axs[0,0])

        axs[0, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[-1, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[-1, 0].set_xlabel(r'$(h,-h,0)$')
        axs[-1, 1].set_xlabel(r'$(h,-h,0)$')
        if yes:
            axs[1,0].set_ylabel(r'$(k,k,-2k)$')
        if yesJP:
            axs[-1,1].set_ylabel(r'$(h,-h,0)$')

    else:
        dirString = "001"
        plthk0fBZ(axs[0,0])

        axs[0, 0].set_ylabel(r'$(0,k,0)$')
        axs[-1, 0].set_ylabel(r'$(0,k,0)$')
        axs[-1, 0].set_xlabel(r'$(h,0,0)$')
        axs[-1, 1].set_xlabel(r'$(h,0,0)$')
        if yes:
            axs[1,0].set_ylabel(r'$(0,k,0)$')
        if yesJP:
            axs[-1,1].set_ylabel(r'$(h,0,0)$')

    htoprint = np.around(2*h,decimals=1)/2

    if len(htoprint) == 3:
        fig.text(1, 0.85, r'$h/J_{yy}=' + str(htoprint[0,0]) + '$', ha='right', va='center', rotation=270)
        fig.text(1, 0.52, r'$h/J_{yy}=' + str(htoprint[0,1]) + '$', ha='right', va='center', rotation=270)
        fig.text(1, 0.2, r'$h/J_{yy}=' + str(htoprint[0,2]) + '$', ha='right', va='center', rotation=270)
    else:
        fig.text(1, 0.75, r'$h/J_{yy}=' + str(htoprint[0,0]) + '$', ha='right', va='center', rotation=270)
        fig.text(1, 0.3, r'$h/J_{yy}=' + str(htoprint[0,1]) + '$', ha='right', va='center', rotation=270)
    for i in range(len(tograph)):
        if "global" in tograph[i]:
            globalS = ''
        else:
            globalS = '\mathrm{LF},'
        temp = tograph[i].split("/")
        JPt = str(JP[i])
        if JP[i] == -0.289:
            JPt = str(-0.3)
        if not tol:
            if (n==h111).all():
                axs[0, i].set_xlabel(
                    r'$\mathcal{S}^{zz}_{' + globalS + '\mathrm{'+ temp[-1][0:-4] + '}}\quad J_\pm=' + JPt + '$',
                    labelpad=20)
                axs[0, i].xaxis.set_label_position('top')
            else:
                axs[0, i].set_xlabel(r'$\mathcal{S}^{zz}_{'+globalS+temp[-1][0:-4]+'}\quad J_\pm='+JPt+'$', labelpad=20)
                axs[0, i].xaxis.set_label_position('top')
        else:
            if (n==h111).all():
                axs[0, i].set_xlabel(
                    r'$J_\pm=' + JPt + '$',
                    labelpad=20)
                axs[0, i].xaxis.set_label_position('top')
            else:
                axs[0, i].set_xlabel(r'$J_\pm='+JPt+'$', labelpad=20)
                axs[0, i].xaxis.set_label_position('top')

    for i in range(h.shape[1]):
        for j in range(len(JP)):
            # print(flux.shape)
            fluxString = flux[j,i]
            try:
                if "global" in tograph[j]:
                    dmax = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm="+str(JP[j])+"_" + fluxString + "/h_" + dirString + "/h=" + str(
                    h[j,i]) + "/Szzglobal.txt" )
                else:
                    dmax = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm="+str(JP[j])+"_" + fluxString + "/h_" + dirString + "/h=" + str(
                    h[j,i]) + "/Szz.txt")
                dmax = np.max(dmax)
                d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm="+str(JP[j])+"_" + fluxString + "/h_" + dirString + "/h=" + str(
                    h[j,i]) + "/" + tograph[j])
                SSSFgenhelper(d /dmax, dirString, axs[i,j], fig)
                if (np.mean(d[-10:,0:20])-np.min(d))/(np.max(d)-np.min(d))>0.5:
                    color = 'black'
                else:
                    color = 'white'
                axs[i,j].text(.01, .99, r"$(\mathrm{"+chr(97+i*len(JP)+j)+"})\;"+fluxStringS[j,i]+"$", ha='left', va='top', transform=axs[i,j].transAxes, color=color)
            except:
                axs[-1, -1].axis('off')
    plt.savefig(filename)
    plt.clf()


def phaseRegraphHelp(d, ax, fig, cb=False):
    c = ax.imshow(d.T, interpolation="nearest", origin='lower', extent=[-0.5, 0.1, 0, 1], aspect='auto')
    if cb:
        cb = fig.colorbar(c, ax=ax)
def phaseExGraph(filename):
    mpl.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(16, 8),layout="constrained")

    axs[0,0].set_xlabel(r'$\mathbf{B}\parallel[110]$', labelpad=10)
    axs[0,0].xaxis.set_label_position('top')
    axs[0,1].set_xlabel(r'$\mathbf{B}\parallel[111]$', labelpad=10)
    axs[0,1].xaxis.set_label_position('top')
    axs[0,2].set_xlabel(r'$\mathbf{B}\parallel[001]$', labelpad=10)
    axs[0,2].xaxis.set_label_position('top')
    axs[1,1].set_xlabel(r'$J_\pm/J_{yy}$', labelpad=10)
    axs[0,0].set_ylabel(r'$h/J_{yy}$')
    axs[1, 0].set_ylabel(r'$h/J_{yy}$')
    d1 = np.loadtxt("../Data/phase_diagrams_exb/phase_110_kappa=2_ex.txt")
    phaseRegraphHelp(d1, axs[0,0], fig)
    d2 = np.loadtxt("../Data/phase_diagrams_exb/phase_110_kappa=2_ex_N.txt")
    phaseRegraphHelp(d2, axs[1,0], fig, True)
    d3 = np.loadtxt("../Data/phase_diagrams_exb/phase_111_kappa=2_ex.txt")
    phaseRegraphHelp(d3, axs[0,1], fig)
    d4 = np.loadtxt("../Data/phase_diagrams_exb/phase_111_kappa=2_ex_N.txt")
    phaseRegraphHelp(d4, axs[1,1], fig, True)
    d5 = np.loadtxt("../Data/phase_diagrams_exb/phase_100_kappa=2_ex.txt")
    phaseRegraphHelp(d5, axs[0,2], fig)
    d6 = np.loadtxt("../Data/phase_diagrams_exb/phase_100_kappa=2_ex_N.txt")
    phaseRegraphHelp(d6, axs[1,2], fig, True)
    axs[0,0].text(.01, .99, r"$(\mathrm{a})$", ha='left', va='top',transform=axs[0,0].transAxes, color='black')
    axs[0,1].text(.01, .99, r"$(\mathrm{b})$", ha='left', va='top',transform=axs[0,1].transAxes, color='black')
    axs[0,2].text(.01, .99, r"$(\mathrm{c})$", ha='left', va='top',transform=axs[0,2].transAxes, color='black')
    axs[1,0].text(.01, .99, r"$(\mathrm{d})$", ha='left', va='top',transform=axs[1,0].transAxes, color='black')
    axs[1,1].text(.01, .99, r"$(\mathrm{e})$", ha='left', va='top',transform=axs[1,1].transAxes, color='black')
    axs[1,2].text(.01, .99, r"$(\mathrm{f})$", ha='left', va='top',transform=axs[1,2].transAxes, color='black')
    plt.savefig(filename)
# phaseExGraph("phase_diagrams_exb.pdf")
# mpl.rcParams.update({'font.size': 15})
# fig, ax = plt.subplots(layout='tight')
# # # # #
# d1 = np.loadtxt("../Data/phase_diagrams/phase_110_kappa=2_mag.txt")
# d1 = d1/4
# h = np.linspace(0,2.2,len(d1.T))
# h = h[15:30]
# d1 = d1[:, 15:30]
# ax.plot(h, d1[200],h, d1[210],h, d1[220],h, d1[230])
# ax.set_xlabel(r"$h/J_{yy}$")
# ax.set_ylabel(r"$|\mathbf{m}|$")
# ax.legend([r'$J_\pm=-0.1$',r'$J_\pm=-0.08$',r'$J_\pm=-0.06$',r'$J_\pm=-0.04$'])
# plt.show()
# DSSF_line_pedantic(20, -0.055, -0.055, 1, 0, 0.1, 2, h110, np.zeros(4), 5, "Files/DSSF/Jpm/")

# ffact = contract('ik, jk->ij', np.array([[0,0,1]]), NN)
# ffact = np.exp(1j * ffact)
# zmag = contract('k,ik->i', h110, z)
# A_pi_here = np.array([[0, 0, 0, 0],
#                       [0, np.pi, np.pi, 0]])
# print(ffact, zmag)z
# print('\n')
# print(np.exp(1j*A_pi_here))

# A = np.loadtxt("../../Data_Archive/phase_110/phase_110_kappa=2_Jpmpm=0.2.txt")
# A = np.where(A==1, np.nan, A)
# A = np.where(A==6, np.nan, A)
# A = np.where(A==11, np.nan, A)
# A = np.where(A==16, np.nan, A)
# C = plt.imshow(A.T, origin='lower', aspect='auto', extent=[-0.3,0.1,0,0.5])
# plt.colorbar(C)
# plt.xlabel(r'$J_\pm/J_{y}$')
# plt.ylabel(r'$h/J_{y}$')
# # plt.show()
# plt.savefig('phase_110_kappa=2_Jpmpm=0.2.png')
# plt.clf()

Jpm = -0.02
Jpmpm = 0
# # #
# Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm),  1.,        2*(Jpmpm-Jpm)
# # # Jxx, Jyy, Jzz = 0.5,     1,         0
# # # # Jxx, Jyy, Jzz = 1,  0.4,         0.2
# fig, axs = plt.subplots()
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, 0.1, flux=np.zeros(4), h=0.1, n=h001, simplified=True)
# a.solvemeanfield()
# # a.graph_loweredge(False,axs,'b')
# # a.graph_upperedge(True,axs,'b')
# A = a.MFE()
# AC = a.condensed
# print(a.chi, a.xi, a.magnetization(), a.gap(), a.MFE())
# # fig, axs = plt.subplots()
# #2420158631264392
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, 0.1, flux=np.ones(4)*np.pi, h=0.1, n=h001, simplified=True)
# a.solvemeanfield()
# # a.graph_loweredge(False,axs,'b')
# # a.graph_upperedge(True,axs,'b')
# A = a.MFE()
# AC = a.condensed
# print(a.chi, a.xi,a.magnetization(), a.gap(), a.MFE())
# a.graph(axs)
# axs.set_ylim([0,0.7])
# plt.show()
#

Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm),  1.,        2*(Jpmpm-Jpm)
Jxx, Jyy, Jzz = 0, 1, 1
fig, axs = plt.subplots()
a = pycon.piFluxSolver(Jxx,Jyy, Jzz, 1, flux=np.zeros(4) * np.pi, h=0.0, n=h111,simplified=False)
a.solvemeanfield(False)
a.graph(axs)
A = a.MFE()
AC = a.condensed
plt.show()
print(a.chi, a.xi, a.magnetization(),a.gap(), a.MFE())
# #
# # Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm),  1.,        2*(Jpmpm-Jpm)
# # fig, axs = plt.subplots()
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=np.zeros(4) * np.pi, h=0.0, n=h111,simplified=False)
# a.solvemeanfield()
# # a.graph(axs)
# B = a.MFE()
# BC = a.condensed
# # plt.show()
# print(a.chi, a.xi, a.magnetization(),a.gap(), a.MFE())



def FFFluxGen(flux):
    return np.array([3*flux, flux, flux, flux])

def nS1helper(filename1, filename2, filename3, fileout):
    nznzMFE = np.loadtxt(filename1+"_MFE.txt")
    z1z1MFE = np.loadtxt(filename2+"_MFE.txt")
    nzz1MFE = np.loadtxt(filename3+"_MFE.txt")
    # nzz1MFE = np.flip(nzz1MFE, axis=0)
    z1nzMFE = np.transpose(nzz1MFE)

    MFE = np.block([[nznzMFE, z1nzMFE],
                    [nzz1MFE, z1z1MFE]])
    np.savetxt(fileout+"_MFE.txt",MFE)
    plt.pcolormesh(MFE)
    plt.savefig(fileout+"_MFE.png")
    plt.clf()

    nznzMFE = np.loadtxt(filename1+".txt")
    z1z1MFE = np.loadtxt(filename2+".txt")
    nzz1MFE = np.loadtxt(filename3+".txt")
    # nzz1MFE = np.flip(nzz1MFE, axis=0)
    z1nzMFE = np.transpose(nzz1MFE)
    MFE = np.block([[nznzMFE, z1nzMFE],
                    [nzz1MFE, z1z1MFE]])
    np.savetxt(fileout+".txt",MFE)
    plt.pcolormesh(MFE)
    plt.savefig(fileout+".png")
    plt.clf()

dir = "../../Data_Archive/phase_XYZ_0_field_0_flux/phase_XYZ_0_field_0_flux_nS=1"
# dir1 = "../../Data_Archive/phase_XYZ_0_field_0_flux/phase_XYZ_0_field_0_flux_ns=1"
#
# nS1helper(dir+"_-10-10", dir+"_0101", dir+"_01-10", dir)
#
# fig, axs = plt.subplots()
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=FFFluxGen(np.pi/3), h=0.2, n=h111,FF=True)
# a.solvemeanfield()
# a.graph(axs)
# E = a.MFE()
# EC = a.condensed
# plt.show()
# print(a.MFE())

# print(A, B, C)
# print(AC, BC, CC)
#

#
# # Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm),  1.,        2*(Jpmpm-Jpm)
# fig, axs = plt.subplots()
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=np.zeros(4) * np.pi, h=0.2, n=h111)
# a.solvemeanfield()
# # a.graph(axs)
# print(a.chi, a.xi, a.magnetization(), a.gap(), a.MFE())
# C = a.MFE()
# CC = a.condensed
#
#
# # Jpmpm=0.2
# # # Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm),  1.,        2*(Jpmpm-Jpm)
# fig, axs = plt.subplots()
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, 1, flux=np.ones(4) * np.pi, h=0.2, n=h111)
# a.solvemeanfield()
# # a.graph(axs)
# D = a.MFE()
# DC = a.condensed
#
# # a.graph(axs)
# print(a.chi, a.xi, a.magnetization(), a.gap(), a.MFE())
#
# # plt.show()
# print(A, B, C, D)
# print(AC, BC, CC, DC)
# print(C,D,E)
# print(CC,DC,EC)
# print(A, C)
# conclude_XYZ_0_field("../Data/phase_diagrams/phase_XYZ_0_field")
# conclude_XYZ_0_field("Misc/New folder/phase_XYZ_0_field")
# conclude_XYZ_0_field("../../Data_Archive/pyrochlore_gmft/phase_XYZ_low_res/phase_XYZ_0_field")



# Jxx = np.linspace(0,0.5,10)
# for i in range(10):
#     SSSF_pedantic(100, Jxx[i], 1, 1-Jxx[i], 0.1, h110, np.ones(4)*np.pi, 30, "Files/XYZ/Jpm=0.25_Jpmpm="+str(1-2*Jxx[i]), "hnhl", K=0, Hr=2.5, Lr=2.5)


# 0.5811437155148994
# [0.64571524 0.64571524] True
# 0.7300161999813554
# emin, emax = a.graph_loweredge(False, axs), a.graph_upperedge(False, axs)
#
# d0 = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_110/h=0.09999999999999999/Szz/10.txt")
# d1 = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_110/h=0.09999999999999999/Szz/11.txt")
# d2 = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_110/h=0.09999999999999999/Szz/12.txt")
# d3 = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_110/h=0.09999999999999999/Szz/13.txt")
#
# d = d0+d1+d2+d3
# c = axs.imshow(d.T/np.max(d))
# fig.colorbar(c)
# plt.show()
# mpl.rcParams.update({'font.size': 25})
# fig, axs = plt.subplots()
# Jpm=0.0
# flux = np.zeros(4)
# # flux = np.ones(4)*np.pi
# # flux = np.array([0,0,np.pi,np.pi])
# a = pycon.piFluxSolver(-2*Jpm,1,-2*Jpm,flux=flux, h=0.2, n=h111)
# a.solvemeanfield()
# print(a.magnetization(), a.GS())
# a.graph(axs)
# plt.show()
#
# fig, axs = plt.subplots()
# mpl.rcParams.update({'font.size': 25})
# fig, axs = plt.subplots(layout='tight',figsize=(12,10))
# axs.set_ylabel(r'$(k,k,-2k)$')
# axs.set_xlabel(r'$(h,-h,0)$')
# h = 0.16
# n=h111
# if not (n==h001).all():
#     htoprint = np.around(2*h,decimals=1)/2
# fig.text(1, 0.5, r'$h/J_{yy}=' + str(htoprint) + '$', ha='right', va='center', rotation=270)
#
#
# d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_111/h=0.16/Szzglobal.txt")
# SSSFgenhelper(d /np.max(d), "111", axs, fig)
# if (np.mean(d[-10:,0:20])-np.min(d))/(np.max(d)-np.min(d))>0.5:
#     color = 'black'
# else:
#     color = 'white'
#
# plt.savefig("h111_sublattice_select2.pdf")

# a = pycon.piFluxSolver(-0.04,1,-0.04,flux=np.array([np.pi,np.pi,np.pi,np.pi]), h=0.3)
# a.solvemeanfield()
# a.graph(axs)
# plt.show()
# mpl.rcParams.update({'font.size': 25})
# fig, axs=plt.subplots(layout='tight',figsize=(8,7))
# d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_111/h=0.16/Szzglobal.txt")
# SSSFgenhelper(d / np.max(d), "111", axs, fig)
# axs.set_xlabel(r"$(h,-h,0)$")
# axs.set_ylabel(r"$(k,k,-2k)$")
# plt.savefig("h111_example.pdf")
# DSSFgraphGen_0(h110,"DSSF_0_field.pdf")
# DSSFgraphGen(0.39999999999999997,0.39999999999999997,np.array([0,0,np.pi,np.pi]), np.nan,h110,"h110_DSSF.pdf")
# DSSFgraphGen(0.2,0.1,np.array([np.pi,np.pi,np.pi,np.pi]),np.nan,h001,"h001_DSSF.pdf")
# DSSFgraphGen(0.3,0.2,np.array([np.pi,np.pi,np.pi,np.pi]), np.nan, h111,"h111_DSSF.pdf")



# SSSFgraphGen(np.array([0,0.18,0.42]),np.array([0,0.18,0.42]), np.array(["pi","pi","00pp"]),np.array([0,0.18]),h110,"Szzglobal.txt", "h110_SSSF.pdf", np.array(['w','b','b','w','b','w','w','w','w','w']))
# SSSFgraphGen(np.array([0,0.16,0.32]),np.array([0,0.2,0.32]), np.array(["pi","pi","pi"]),np.array([0,0.2,0.32]),h111,"Szzglobal.txt", "h111_SSSF.pdf", np.array(['w','w','w','w','w','w','w','w','w','w']))
# SSSFgraphGen(np.array([0,0.08,0.2]),np.array([0,0.08,0.2]), np.array(["pi","pi","0"]),np.array([0,0.08]),h001,"Szzglobal.txt", "h001_SSSF.pdf", np.array(['b','w','w','b','w','w','w','w','w','w']))

# sublatticeSSSFgraphGen(h110, np.array(["Szz/00.txt", "Szz/12.txt", "Szzglobal/03.txt", "Szzglobal/12.txt"]), np.array([0,0.18,0.42]), np.array([0.03,-0.03]), np.array([["0","0","0"],["pi","pi","00pp"]]), "Misc/h110_sublattice.pdf")
# sublatticeSSSFgraphGen(h111, np.array(["Szz/Kagome.txt", "Szz/Triangular.txt", "Szzglobal/Kagome.txt", "Szz/Kagome-Tri.txt"]), np.array([0,0.2,0.32]), np.array([0.03,-0.289]), np.array([["0","0","0"],["pi","pi","pi"]]), "Misc/h111_sublattice.pdf")
# sublatticeSSSFgraphGen(h001, np.array(["Szz/01.txt", "Szz/03.txt", "Szzglobal/01.txt", "Szzglobal/03.txt"]), np.array([0,0.04,0.2]), np.array([0.03,-0.03]), np.array([["0","0","0"],["pi","pi","0"]]), "Misc/h001_sublattice.pdf")
# sublatticeSSSFgraphGen_select(h110, np.array(["Szzglobal.txt", "Szzglobal.txt"]), np.array([[0.0,0.42],[0.0,0.42]]), np.array([0.03,-0.03]), np.array([["0","0"],["pi","00pp"]]), np.array([["0$-$\mathrm{flux}","0$-$\mathrm{flux}"],["\pi$-$\mathrm{flux}","(0,\pi,\pi,0)"]]),
#                               "Misc/h110_SSSF.pdf", tol=True)
# sublatticeSSSFgraphGen_select(h110, np.array(["Szzglobal.txt", "Szzglobal.txt", "Szzglobal.txt"]), np.array([[0.12,0],[0.42,0],[0.42,0]]), np.array([-0.289,-0.03,0.03]), np.array([["pi","pi"],["00pp","pi"],["0","0"]]), np.array([["\pi$-$\mathrm{flux}","\pi$-$\mathrm{flux}"],["(0,\pi,\pi,0)","\pi$-$\mathrm{flux}"],["0$-$\mathrm{flux}","0$-$\mathrm{flux}"]]),
#                               "h110_SSSF.pdf", tol=True)
# sublatticeSSSFgraphGen_select(h110, np.array(["Szz/00.txt", "Szz/00.txt", "Szz/00.txt"]), np.array([[0.18,0.18],[0.42,0.42],[0.42,0.42]]), np.array([-0.289,-0.03,0.03]), np.array([["pi","pi"],["pi","00pp"],["0","0"]]), np.array([["\pi$-$\mathrm{flux}","\pi$-$\mathrm{flux}"],["\pi$-$\mathrm{flux}","(0,\pi,\pi,0)"],["0$-$\mathrm{flux}","0$-$\mathrm{flux}"]]),
#                               "h110_SSSF_1.pdf", tol=True)
# sublatticeSSSFgraphGen_select(h111, np.array(["Szzglobal.txt", "Szzglobal.txt"]), np.array([[0.0,0.32],[0.0,0.32]]), np.array([0.03,-0.03]), np.array([["0","0"],["pi","pi"]]), np.array([["0$-$\mathrm{flux}","0$-$\mathrm{flux}"],["\pi$-$\mathrm{flux}","\pi$-$\mathrm{flux}"]]),
#                               "Misc/h111_SSSF.pdf",  tol=True)
# sublatticeSSSFgraphGen_select(h001, np.array(["Szzglobal.txt", "Szzglobal.txt"]), np.array([[0,0.2],[0,0.2]]), np.array([0.03,-0.03]), np.array([["0","0"],["pi","0"]]), np.array([["0$-$\mathrm{flux}","0$-$\mathrm{flux}"],["\pi$-$\mathrm{flux}","0$-$\mathrm{flux}"]]),
#                               "Misc/h001_SSSF.pdf",  tol=True)
# #
# sublatticeSSSFgraphGen_select(h110, np.array(["Szzglobal/03.txt", "Szzglobal/03.txt"]), np.array([[0.0,0.18],[0.0,0.18]]), np.array([0.03,-0.03]), np.array([["0","0"],["pi","pi"]]), np.array([["0$-$\mathrm{flux}","0$-$\mathrm{flux}"],["\pi$-$\mathrm{flux}","\pi$-$\mathrm{flux}"]]),
#                               "Misc/h110_sublattice_select1.pdf")
# #
# sublatticeSSSFgraphGen_select(h110, np.array(["Szz/12.txt", "Szz/12.txt"]), np.array([[0.18,0.42],[0.18,0.42]]), np.array([0.03,-0.03]), np.array([["0","0"],["pi","00pp"]]), np.array([["0$-$\mathrm{flux}","0$-$\mathrm{flux}"],["\pi$-$\mathrm{flux}","(0,\pi,\pi,0)"]]),
#                               "Misc/h110_sublattice_select3.pdf", 1)
# sublatticeSSSFgraphGen_select(h110, np.array(["Szz/00.txt", "Szz/00.txt"]), np.array([[0.18,0.42],[0.18,0.42]]), np.array([-0.03,-0.289]), np.array([["pi","00pp"],["pi","pi"]]), np.array([["\pi$-$\mathrm{flux}","(0,\pi,\pi,0)"],["\pi$-$\mathrm{flux}","\pi$-$\mathrm{flux}"]]),
#                               "Misc/h110_sublattice_select2.pdf")
# #
# sublatticeSSSFgraphGen_select(h111, np.array(["Szz/Kagome-Tri.txt", "Szzglobal/Kagome.txt"]), np.array([[0.2,0.32],[0.2,0.32]]), np.array([0.03,-0.289]), np.array([["0","0"],["pi","pi"]]), np.array([["0$-$\mathrm{flux}","0$-$\mathrm{flux}"],["\pi$-$\mathrm{flux}","\pi$-$\mathrm{flux}"]]),
#                               "Misc/h111_sublattice_select.pdf")
# sublatticeSSSFgraphGen_select(h001, np.array(["Szz/01.txt", "Szz/03.txt"]), np.array([0.08,0.2]), np.array([0.03,-0.03]), np.array([["0","0"],["pi","0"]]), np.array([["0$-$\mathrm{flux}","0$-$\mathrm{flux}"],["\pi$-$\mathrm{flux}","0$-$\mathrm{flux}"]]),
#                               "Misc/h001_sublattice_select.pdf")
# sublatticeSSSFgraphGen_select(h111, np.array(["Szzglobal.txt"]), np.array([0.16]), np.array([-0.289]), np.array([["pi"]]), np.array([["\pi$-$\mathrm{flux}"]]), "h111_sublattice_select2.pdf")

# e = np.linspace(0, 2.5,200)
# A = graph_2S_rho(e, np.array([-0.1]), 0, h110, np.ones(4)*np.pi, 10, 1e-3, 0, 1)
# plt.plot(e, A.T)
# plt.savefig('test.pdf')
# d = np.loadtxt("testtwo_spinon_DOS_111.txt")
# plt.imshow(d.T, interpolation="lanczos", origin='lower', extent=[0, 0.5, 0, 2], aspect='auto', cmap='gnuplot')
# plt.ylabel(r'$\omega/J_{yy}$')
# plt.xlabel(r'$h/J_{yy}$')
# # plt.colorbar()
# plt.savefig("testtwo_spinon_DOS_111.pdf")
# from scipy.ndimage.filters import gaussian_filter
#

# TwoSpinonDOS_111(50, 30, "two_spinon_DOS_111_Jpm=-0.03.pdf")
# plt.clf()
# #
# TwoSpinonDOS_001(50, 30, "two_spinon_DOS_001_Jpm=-0.03.pdf")
# plt.clf()
# #
# TwoSpinonDOS_110(50, 30, "two_spinon_DOS_110_Jpm=-0.03.pdf")
# plt.clf()
# # mpl.rcParams.update({'font.size': 25})
#
# TwoSpinonDOS_110_a(50, 30, "two_spinon_DOS_110_Jpm=-0.3.pdf")
# plt.clf()
# # #
# TwoSpinonDOS_001_a(50, 30, "two_spinon_DOS_001_Jpm=-0.3.pdf")
# plt.clf()
# TwoSpinonDOS_111_a(50, 30, "two_spinon_DOS_111_Jpm=0.03.pdf")
# plt.clf()
# mpl.rcParams.update({'font.size': 15})
# fig, axs = plt.subplots()
# a = np.loadtxt("two_spinon_DOS_111_Jpm=0.03.pdf.txt")
# c = axs.imshow(a.T, interpolation="gaussian", origin='lower', extent=[0, 0.4, 0.5, 1.8], aspect='auto', cmap='gnuplot')
# cb = fig.colorbar(c, ax=axs)
# axs.set_ylabel(r'$\omega/J_{yy}$')
# axs.set_xlabel(r'$h/J_{yy}$')
# plt.savefig("two_spinon_DOS_111_Jpm=0.03.pdf")


# plt.clf()
# # #
# mpl.rcParams.update({'font.size': 25})
# fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(20,10),layout='tight')
#
# a = np.loadtxt("two_spinon_DOS_110_Jpm=-0.03.pdf.txt")
# c = axs[0,0].imshow(a.T, interpolation="gaussian", origin='lower', extent=[0, 0.3, 0.44, 1.6], aspect='auto', cmap='gnuplot')
# cb = fig.colorbar(c, ax=axs[0,0])
# axs[0,0].set_ylabel(r'$\omega/J_{yy}$')
# # axs[0,0].set_xlabel(r'$h/J_{yy}$')
# # plt.savefig("two_spinon_DOS_111_Jpm=0.03.pdf")
# axs[0,0].set_xlabel(r'$\mathbf{B}\parallel[110]$', labelpad=10)
# axs[0,0].xaxis.set_label_position('top')
# axs[0,0].axvline(x=0.2, color='w', label='axvline - full height', linestyle='dashed')
# axs[0,0].text(.15, .99, r"$\pi$-$\mathrm{flux}$", ha='left', va='top',transform=axs[0,0].transAxes, color='w')
# axs[0,0].text(.68, .99, r"$(0,\pi,\pi,0)$", ha='left', va='top',transform=axs[0,0].transAxes, color='w')
#
# a = np.loadtxt("two_spinon_DOS_111_Jpm=-0.03.pdf.txt")
# c = axs[0,1].imshow(a.T, interpolation="gaussian", origin='lower', extent=[0, 0.5, 0.4, 1.8], aspect='auto', cmap='gnuplot')
# cb = fig.colorbar(c, ax=axs[0,1])
# axs[0, 1].set_xlabel(r'$\mathbf{B}\parallel[111]$', labelpad=10)
# axs[0, 1].xaxis.set_label_position('top')
# axs[0, 1].axvline(x=0.33, color='w', label='axvline - full height', linestyle='dashed')
# axs[0,1].text(.15, .99, r"$\pi$-$\mathrm{flux}$", ha='left', va='top',transform=axs[0,1].transAxes, color='w')
# axs[0,1].text(.72, .99, r"$0$-$\mathrm{flux}$", ha='left', va='top',transform=axs[0,1].transAxes, color='w')
#
# a = np.loadtxt("two_spinon_DOS_001_Jpm=-0.03.pdf.txt")
# c = axs[0,2].imshow(a.T, interpolation="gaussian", origin='lower', extent=[0, 0.22, 0.6, 1.6], aspect='auto', cmap='gnuplot')
# cb = fig.colorbar(c, ax=axs[0,2])
# axs[0, 2].set_xlabel(r'$\mathbf{B}\parallel[001]$', labelpad=10)
# axs[0, 2].xaxis.set_label_position('top')
# axs[0, 2].axvline(x=0.17, color='w', label='axvline - full height', linestyle='dashed')
# axs[0,2].text(.15, .99, r"$\pi$-$\mathrm{flux}$", ha='left', va='top',transform=axs[0,2].transAxes, color='w')
# axs[0,2].text(.80, .99, r"$0$-$\mathrm{flux}$", ha='left', va='top',transform=axs[0,2].transAxes, color='w')
#
# a = np.loadtxt("two_spinon_DOS_110_Jpm=-0.3.pdf.txt")
# c = axs[1,0].imshow(a.T, interpolation="gaussian", origin='lower', extent=[0, 0.23, 0, 3], aspect='auto', cmap='gnuplot')
# cb = fig.colorbar(c, ax=axs[1,0])
# axs[1,0].set_ylabel(r'$\omega/J_{yy}$')
# axs[1,0].set_xlabel(r'$h/J_{yy}$')
# axs[1,0].text(.15, .99, r"$\pi$-$\mathrm{flux}$", ha='left', va='top',transform=axs[1,0].transAxes, color='w')
#
# a = np.loadtxt("two_spinon_DOS_111_Jpm=-0.3.pdf.txt")
# c = axs[1,1].imshow(a.T, interpolation="gaussian", origin='lower', extent=[0, 0.35, 0, 3], aspect='auto', cmap='gnuplot')
# cb = fig.colorbar(c, ax=axs[1,1])
# axs[1,1].set_xlabel(r'$h/J_{yy}$')
# axs[1,1].text(.15, .99, r"$\pi$-$\mathrm{flux}$", ha='left', va='top',transform=axs[1,1].transAxes, color='w')
#
# a = np.loadtxt("two_spinon_DOS_001_Jpm=-0.3.pdf.txt")
# c = axs[1,2].imshow(a.T, interpolation="gaussian", origin='lower', extent=[0, 0.1, 0, 3], aspect='auto', cmap='gnuplot')
# cb = fig.colorbar(c, ax=axs[1,2])
# axs[1,2].set_xlabel(r'$h/J_{yy}$')
# axs[1,2].text(.15, .99, r"$\pi$-$\mathrm{flux}$", ha='left', va='top',transform=axs[1,2].transAxes, color='w')
#
# fig.text(1, 0.75, r'$J_\pm/J_{yy}=-0.03$', ha='right', va='center', rotation=270)
# fig.text(1, 0.3, r'$J_\pm/J_{yy}=-0.3$', ha='right', va='center', rotation=270)
#
# axs[0,0].text(.01, .99, r"$(\mathrm{a})$", ha='left', va='top',transform=axs[0,0].transAxes, color='w')
# axs[0,1].text(.01, .99, r"$(\mathrm{b})$", ha='left', va='top',transform=axs[0,1].transAxes, color='w')
# axs[0,2].text(.01, .99, r"$(\mathrm{c})$", ha='left', va='top',transform=axs[0,2].transAxes, color='w')
# axs[1,0].text(.01, .99, r"$(\mathrm{d})$", ha='left', va='top',transform=axs[1,0].transAxes, color='w')
# axs[1,1].text(.01, .99, r"$(\mathrm{e})$", ha='left', va='top',transform=axs[1,1].transAxes, color='w')
# axs[1,2].text(.01, .99, r"$(\mathrm{f})$", ha='left', va='top',transform=axs[1,2].transAxes, color='w')
#
# plt.savefig("two_spinon_DOS.pdf")

# mpl.rcParams.update({'font.size': 25})
# fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(20,12),layout='constrained')
# #
# d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_110/h=0.0/Szzglobal.txt")
# if (np.mean(d[-10:, 0:30]) - np.min(d)) / (np.max(d) - np.min(d)) > 0.5:
#     color = 'black'
# else:
#     color = 'white'
# SSSFgenhelper(d / np.max(d), "110", axs[0, 0], fig)
# axs[0,0].set_ylabel(r'$(0,0,l)$')
# axs[0,0].set_xlabel(r'$(h,-h,0)$')
# plthhlfBZ(axs[0,0])
#
# d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_111/h=0.0/Szzglobal.txt")
# if (np.mean(d[-10:, 0:30]) - np.min(d)) / (np.max(d) - np.min(d)) > 0.5:
#     color = 'black'
# else:
#     color = 'white'
# SSSFgenhelper(d / np.max(d), "111", axs[0, 1], fig)
# axs[0,1].set_ylabel(r'$(k,k,-2k)$')
# axs[0,1].set_xlabel(r'$(h,-h,0)$')
# plthh2kfBZ(axs[0,1])
# d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_001/h=0.0/Szzglobal.txt")
# if (np.mean(d[-10:, 0:30]) - np.min(d)) / (np.max(d) - np.min(d)) > 0.5:
#     color = 'black'
# else:
#     color = 'white'
# SSSFgenhelper(d / np.max(d), "001", axs[0, 2], fig)
# axs[0,2].set_ylabel(r'$(0,k,0)$')
# axs[0,2].set_xlabel(r'$(h,0,0)$')
# plthk0fBZ(axs[0,2])
# d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_110/h=0.12/Szzglobal.txt")
# if (np.mean(d[-10:, 0:30]) - np.min(d)) / (np.max(d) - np.min(d)) > 0.5:
#     color = 'black'
# else:
#     color = 'white'
# SSSFgenhelper(d / np.max(d), "110", axs[1, 0], fig)
# axs[1,0].set_ylabel(r'$(0,0,l)$')
# axs[1,0].set_xlabel(r'$(h,-h,0)$')
#
# d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_111/h=0.2/Szzglobal.txt")
# if (np.mean(d[-10:, 0:30]) - np.min(d)) / (np.max(d) - np.min(d)) > 0.5:
#     color = 'black'
# else:
#     color = 'white'
# SSSFgenhelper(d / np.max(d), "111", axs[1, 1], fig)
# axs[1,1].set_ylabel(r'$(k,k,-2k)$')
# axs[1,1].set_xlabel(r'$(h,-h,0)$')
#
# d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_001/h=0.05/Szzglobal.txt")
# if (np.mean(d[-10:, 0:30]) - np.min(d)) / (np.max(d) - np.min(d)) > 0.5:
#     color = 'black'
# else:
#     color = 'white'
# SSSFgenhelper(d / np.max(d), "001", axs[1, 2], fig)
# axs[1,2].set_ylabel(r'$(0,k,0)$')
# axs[1,2].set_xlabel(r'$(h,0,0)$')
#
# axs[0,0].text(.01, .99, r"$(2\mathrm{a})$", ha='left', va='top',transform=axs[0,0].transAxes, color='w')
# axs[0,1].text(.01, .99, r"$(2\mathrm{b})$", ha='left', va='top',transform=axs[0,1].transAxes, color='w')
# axs[0,2].text(.01, .99, r"$(2\mathrm{c})$", ha='left', va='top',transform=axs[0,2].transAxes, color='w')
# axs[1,0].text(.01, .99, r"$(2\mathrm{d})$", ha='left', va='top',transform=axs[1,0].transAxes, color='w')
# axs[1,1].text(.01, .99, r"$(2\mathrm{e})$", ha='left', va='top',transform=axs[1,1].transAxes, color='w')
# axs[1,2].text(.01, .99, r"$(2\mathrm{f})$", ha='left', va='top',transform=axs[1,2].transAxes, color='w')
# plt.savefig("synopsis2.pdf")
# plt.clf()
# #
# mpl.rcParams.update({'font.size': 25})
# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(22,5), layout='tight')
# a = pycon.piFluxSolver(0.289 * 2, 0.289 * 2, 1, flux=np.ones(4) * np.pi, h=0.2, n=h110, simplified=True)
# a.solvemeanfield()
#
# emin, emax = a.graph_loweredge(False, axs[0]), a.graph_upperedge(False, axs[0])
# d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_110/h=0.2/Szzglobal.txt")
# emin, emax = np.min(emin) * 0.95, np.max(emax) * 1.02
# d = DSSFparse(emin, emax, d)
#
# c = axs[0].imshow(d.T / np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, 0, emax],
#                      aspect='auto', cmap='gnuplot2')
# fig.colorbar(c, ax=axs[0])
#
# a = pycon.piFluxSolver(0.289 * 2, 0.289 * 2, 1, flux=np.ones(4) * np.pi, h=0.3, n=h111, simplified=True)
# a.solvemeanfield()
#
# emin, emax = a.graph_loweredge(False, axs[1]), a.graph_upperedge(False, axs[1])
# d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_111/h=0.3/Szzglobal.txt")
# emin, emax = np.min(emin) * 0.95, np.max(emax) * 1.02
# d = DSSFparse(emin, emax, d)
#
# c = axs[1].imshow(d.T / np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, 0, emax],
#                      aspect='auto', cmap='gnuplot2')
#
# fig.colorbar(c, ax=axs[1])
# #
# a = pycon.piFluxSolver(0.289 * 2, 0.289 * 2, 1, flux=np.ones(4) * np.pi, h=0.075, n=h001, simplified=True)
# a.solvemeanfield()
#
# emin, emax = a.graph_loweredge(False, axs[2]), a.graph_upperedge(False, axs[2])
# d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_001/h=0.07500000000000001/Szzglobal.txt")
# emin, emax = np.min(emin) * 0.95, np.max(emax) * 1.02
# d = DSSFparse(emin, emax, d)
# c = axs[2].imshow(d.T / np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, 0, emax],
#                      aspect='auto', cmap='gnuplot2')
# fig.colorbar(c, ax=axs[2])

# # a = np.loadtxt("Misc/two_spinon_DOS_110_Jpm=-0.3.pdf.txt")
# # c = axs[1,0].imshow(a.T, interpolation="gaussian", origin='lower', extent=[0, 0.23, 0, 3], aspect='auto', cmap='gnuplot')
# # cb = fig.colorbar(c, ax=axs[1,0])
# # axs[1,0].set_ylabel(r'$\omega/J_{yy}$')
# # axs[1,0].set_xlabel(r'$h/J_{yy}$')
# #
# # a = np.loadtxt("Misc/two_spinon_DOS_111_Jpm=-0.3.pdf.txt")
# # c = axs[1,1].imshow(a.T, interpolation="gaussian", origin='lower', extent=[0, 0.3, 0, 3], aspect='auto', cmap='gnuplot')
# # cb = fig.colorbar(c, ax=axs[1,1])
# # axs[1,1].set_xlabel(r'$h/J_{yy}$')
# #
# # a = np.loadtxt("Misc/two_spinon_DOS_001_Jpm=-0.3.pdf.txt")
# # c = axs[1,2].imshow(a.T, interpolation="gaussian", origin='lower', extent=[0, 0.08, 0, 3], aspect='auto', cmap='gnuplot')
# # cb = fig.colorbar(c, ax=axs[1,2])
# # axs[1,2].set_xlabel(r'$h/J_{yy}$')
#
# # fig.text(1, 0.75, r'$J_\pm/J_{yy}=-0.03$', ha='right', va='center', rotation=270)
# # fig.text(1, 0.3, r'$J_\pm/J_{yy}=-0.3$', ha='right', va='center', rotation=270)
#
# axs[0].text(.01, .99, r"$(3\mathrm{a})$", ha='left', va='top',transform=axs[0].transAxes, color='w')
# axs[1].text(.01, .99, r"$(3\mathrm{b})$", ha='left', va='top',transform=axs[1].transAxes, color='w')
# axs[2].text(.01, .99, r"$(3\mathrm{c})$", ha='left', va='top',transform=axs[2].transAxes, color='w')
# # # axs[1,0].text(.01, .99, r"$(3\mathrm{d})$", ha='left', va='top',transform=axs[1,0].transAxes, color='w')
# # # axs[1,1].text(.01, .99, r"$(3\mathrm{e})$", ha='left', va='top',transform=axs[1,1].transAxes, color='w')
# # # axs[1,2].text(.01, .99, r"$(3\mathrm{f})$", ha='left', va='top',transform=axs[1,2].transAxes, color='w')
# # #
# plt.savefig("synopsis3.pdf")
# mpl.rcParams.update({'font.size': 20})
# fig,axs = plt.subplots(layout='tight',figsize=(6,5))
# d1 = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_110/h=0.0/Szzglobal.txt")
# d2 = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_110/h=0.12/Szzglobal.txt")
# SSSFgenhelper(d2-d1,"hhl",axs,fig)
# axs.set_xlabel(r"$(h,-h,0)$")
# axs.set_ylabel(r"$(0,0,l)$")
# plt.show()
# TwoSpinonDOS_111_a(50, 10, "test_Jpm")
# d = nc.Dataset("SSSF_April_25/Jpm=-0.03_0/h_001/h=0.0/full_info.nc")
#
# regraphSSSF("SSSF_April_25")
#
# dir = "../Data/phase_diagrams"
# regraphPhase(dir)

# dir = "../Data/Final_SSSF_pedantic"
# regraphSSSF(dir)

#
# dir = "Final/Jpm=-0.03_pi/h_111/"
# regraph(dir, "hkk")
# dir = "Final/Jpm=-0.03_0/h_111/"
# regraph(dir, "hkk")
# dir = "Final/Jpm=0.02_0/h_111/"
# regraph(dir, "hkk")
# dir = "Final/Jpm=0.02_pi/h_111/"
# regraph(dir, "hkk")
# dir = "Final/Jpm=-0.289_pi/h_111/"
# regraph(dir, "hkk")
# dir = "Final/Jpm=-0.289_0/h_111/"
# regraph(dir, "hkk")
#
# dir = "Final/Jpm=-0.03_pi/h_001/"
# regraph(dir, "hk0")
# dir = "Final/Jpm=-0.03_0/h_001/"
# regraph(dir, "hk0")
# dir = "Final/Jpm=0.02_0/h_001/"
# regraph(dir, "hk0")
# dir = "Final/Jpm=0.02_pi/h_001/"
# regraph(dir, "hk0")
# dir = "Final/Jpm=-0.289_pi/h_001/"
# regraph(dir, "hk0")
# dir = "Final/Jpm=-0.289_0/h_001/"
# regraph(dir, "hk0")
#
#
