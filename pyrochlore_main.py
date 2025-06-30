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
mpl.rcParams.update({'font.size': 25})

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
            axs[0, 1].text(.01, .99, r"$(\mathrm{b})\; \pi$-$\mathrm{flux}$", ha='left', va='top', transform=axs[0, 1].transAxes,
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
    d = np.loadtxt("../../Data_Archive/pyrochlore_gmft/Data/Final_DSSF_pedantic/Jpm=0.03_0/h_"+dirString+"/h="+str(h0)+"/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    d = DSSFparse(emin, emax, d)
    c = axs[1,0].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, 0, emax], aspect='auto', cmap='gnuplot2')
    fig.colorbar(c, ax=axs[1,0])

    a = pycon.piFluxSolver(0.06,0.06,1,flux=mid,h=hmid,n=n, simplified=True)
    a.solvemeanfield()
    a.graph(axs[0,1])

    emin, emax = a.graph_loweredge(False, axs[1,1]), a.graph_upperedge(False, axs[1,1])
    d = np.loadtxt("../../Data_Archive/pyrochlore_gmft/Data/Final_DSSF_pedantic/Jpm=-0.03_"+fluxString+"/h_"+dirString+"/h="+str(hmid)+"/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    d = DSSFparse(emin, emax, d)
    c = axs[1,1].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, 0, emax], aspect='auto', cmap='gnuplot2')
    fig.colorbar(c, ax=axs[1,1])

    if not np.isnan(hpi):
        a = pycon.piFluxSolver(0.289*2,0.289*2,1,flux=np.ones(4)*np.pi,h=hpi,n=n, simplified=True)
        a.solvemeanfield()
        a.graph(axs[0,2])

        emin, emax = a.graph_loweredge(False, axs[1,2]), a.graph_upperedge(False, axs[1,2])
        d = np.loadtxt("../../Data_Archive/pyrochlore_gmft/Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_"+dirString+"/h="+str(hpi)+"/Szzglobal.txt")
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

def nancount(i, j, A):
    count = 0
    if np.isnan(A[i][j+1]):
        count = count + 1
    if np.isnan(A[i][j-1]):
        count = count + 1
    if np.isnan(A[i+1][j]):
        count = count + 1
    if np.isnan(A[i-1][j]):
        count = count + 1
    if np.isnan(A[i+1][j + 1]):
        count = count + 1
    if np.isnan(A[i-1][j - 1]):
        count = count + 1
    if np.isnan(A[i + 1][j+1]):
        count = count + 1
    if np.isnan(A[i - 1][j+1]):
        count = count + 1

    if count >= 6:
        return True
    else:
        return False

def smooth(A):
    for i in range(1, len(A) - 1):
        for j in range(1, len(A.T) - 1):
            if nancount(i, j, A):
                A[i, j] = np.nan
    return A

