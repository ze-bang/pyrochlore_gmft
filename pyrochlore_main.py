import os
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['cm']
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

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
def SSSFgenhelper(d, hhl, ax, fig, showBZ=False):

    if hhl == "110":
        c = ax.imshow(d, interpolation="lanczos", origin='lower', extent=[-2.5, 2.5, -2.5, 2.5], aspect='equal')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        if showBZ:
            Gamms = np.array([[0,0],[1,1],[-1,1],[1,-1],[-1,-1],[2,0],[0,2],[-2,0],[0,-2],[2,2],[-2,2],[2,-2],[-2,-2]])
            Ls = np.array([[0.5,0.5]])
            Xs = np.array([[0,1]])
            Us = np.array([[0.25,1]])
            Ks = np.array([[0.75,0]])


            Boundary = np.array([[0.25, 1],[-0.25,1],[-0.75,0],[-0.25,-1],[0.25,-1],[0.75,0]])

            plot_BZ_hhl(Gamms[0], Boundary, 'w:',ax)

            ax.scatter(Gamms[0,0], Gamms[0,1],zorder=6)
            ax.scatter(Ls[:,0], Ls[:,1],zorder=6)
            ax.scatter(Xs[:, 0], Xs[:, 1],zorder=6)
            ax.scatter(Ks[:, 0], Ks[:, 1],zorder=6)
            ax.scatter(Us[:, 0], Us[:, 1],zorder=6)
            plot_text(Gamms,r'$\Gamma$',ax)
            plot_text(Ls,r'$L$',ax)
            plot_text(Xs,r'$X$',ax, offset_adjust=np.array([-0.3,0]))
            plot_text(Us,r'$U$',ax)
            plot_text(Ks,r'$K$',ax)

    elif hhl == "111":
        c = ax.imshow(d, interpolation="lanczos", origin='lower', extent=[-3, 3, -3, 3], aspect='auto')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
    else:
        c = ax.imshow(d, interpolation="lanczos", origin='lower', extent=[-2.5, 2.5, -2.5, 2.5], aspect='auto')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
    return c

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
# print("What the fuck is going on")
# mpl.rcParams.update({'font.size': 20})
#
# fig, ax = plt.subplots(1,3,figsize=(16, 4))
#
# A = np.abs(np.loadtxt("phase_100_kappa=2_octupolar_mag.txt"))/2
# Jpm = np.linspace(-0.5, 0.1, 300)
# h = np.linspace(0, 0.5, 150)
# # plt.imshow(A.T, origin='lower', extent=[-0.5, 0.1, 0.0, 0.5], aspect='auto')
# C = ax[2].pcolormesh(Jpm, h, A.T)
# ax[2].set_xticks(np.arange(-0.5, 0.1, step=0.1))
# fig.colorbar(C, ax=ax[2])
# ax[2].set_xlabel(r"$J_\pm/J_{yy}$")
# ax[2].text(.05,.9,r'$(\mathrm{c})$',
#         transform=ax[2].transAxes)
#
# A = np.abs(np.loadtxt("phase_110_kappa=2_octupolar_mag.txt"))/2
# A[252:-1, 110:-1] = np.nan
# A = smooth(smooth(smooth(smooth(smooth(smooth(A))))))
#
# Jpm = np.linspace(-0.5, 0.1, 300)
# h = np.linspace(0, 2.2, 150)
# # plt.imshow(A.T, origin='lower', extent=[-0.5, 0.1, 0.0, 2.2], aspect='auto')
# C = ax[0].pcolormesh(Jpm, h, A.T)
# ax[0].set_xticks(np.arange(-0.5, 0.1, step=0.1))
# fig.colorbar(C, ax=ax[0])
# ax[0].set_xlabel(r"$J_\pm/J_{yy}$")
# ax[0].text(.05,.9,r'$(\mathrm{a})$',
#         transform=ax[0].transAxes)
# ax[0].set_ylabel(r"$h/J_{yy}$")
# A = np.abs(np.loadtxt("phase_111_kappa=2_octupolar_mag.txt"))/2
#
# A = smooth(smooth(smooth(smooth(smooth(smooth(A))))))
#
# Jpm = np.linspace(-0.5, 0.1, 300)
# h = np.linspace(0, 0.7, 150)
#
# for i in range(len(A)):
#     for j in range(len(A.T)):
#         Jpm_here = Jpm[i]
#         h_here = h[j]
#         if h_here > np.log(Jpm_here+1.67) and Jpm_here<-0.333 and Jpm_here > -0.45:
#             A[i,j] = np.nan
#
# # plt.imshow(A.T, origin='lower', extent=[-0.5, 0.1, 0.0, 0.7], aspect='auto')
# C = ax[1].pcolormesh(Jpm, h, A.T)
# ax[1].set_xticks(np.arange(-0.5, 0.1, step=0.1))
# fig.colorbar(C, ax=ax[1])
# ax[1].set_xlabel(r"$J_\pm/J_{yy}$")
# ax[1].text(.05,.9,r'$(\mathrm{b})$',
#         transform=ax[1].transAxes)
#
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
# # phaseExGraph("phase_diagrams_exb.pdf")
# mpl.rcParams.update({'font.size': 8})
# axinset = inset_axes(ax[0], width="100%", height="100%",bbox_to_anchor=(0.19, 0.3, 0.45,0.5), loc=3, bbox_transform=ax[0].transAxes)
# # # # # # #
# d1 = np.loadtxt("phase_110_kappa=2_octupolar_mag.txt")
# d1 = d1/2
# h = np.linspace(0,2.2,len(d1.T))
# dt = 2.2/len(d1.T)
# h = h[10:25]
# d1 = d1[:, 10:25]
# # ax.plot(h, np.diff(d1[200])/dt,h, np.diff(d1[210])/dt,h, np.diff(d1[220])/dt,h, np.diff(d1[230])/dt)
# axinset.plot(h, d1[200],h, d1[210],h, d1[220],h, d1[230])
# axinset.set_xlabel(r"$h/J_{yy}$")
# axinset.set_ylabel(r"$|\mathbf{m}|$")
# axinset.legend([r'$J_\pm=-0.1$',r'$J_\pm=-0.08$',r'$J_\pm=-0.06$',r'$J_\pm=-0.04$'], prop={'size': 6}, loc=4)
#
#
#
# fig.tight_layout()
# bbox = axinset.get_tightbbox(fig.canvas.get_renderer())
# x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
# # slightly increase the very tight bounds:
# xpad = 0.05 * width
# ypad = 0.05 * height
# fig.add_artist(plt.Rectangle((x0 - xpad, y0 - ypad), width + 2 * xpad, height + 2 * ypad, edgecolor='black', linewidth=1,
#                              fill=False))
# plt.savefig("magnetization.pdf")
# # plt.clf()
# # DSSF_line_pedantic(20, -0.055, -0.055, 1, 0, 0.1, 2, h110, np.zeros(4), 5, "Files/DSSF/Jpm/")
# plt.show()
# ffact = contract('ik, jk->ij', np.array([[0,0,1]]), NN)
# ffact = np.exp(1j * ffact)
# zmag = contract('k,ik->i', h110, z)
# A_pi_here = np.array([[0, 0, 0, 0],
#                       [0, np.pi, np.pi, 0]])
# print(ffact, zmag)z
# print('\n')
# print(np.exp(1j*A_pi_here))

# A = np.loadtxt("../../Data_Archive/pyrochlore_gmft/phase_110/phase_110_kappa=2_Jpmpm=0.2.txt")
# A = np.where(A==1, np.nan, A)
# A = np.where(A==6, np.nan, A)
# A = np.where(A==11, np.nan, A)
# A = np.where(A==16, np.nan, A)
# print(np.arange(10)[1::2])

# A = np.abs(np.loadtxt("MC_phase_diagram_CZO_001_phase.txt"))
# plt.imshow(A.T, extent=[-0.3, 0.3, 0, 3.0], origin="lower", aspect="auto")
# plt.xlabel(r"$J_\pm/J_{yy}$")
# plt.ylabel(r"$h/J_{yy}$")
# plt.title("Magnetization")
# plt.colorbar()
# plt.show()
# #
# A_x, A_y = np.gradient(A)
# # plt.imshow(A_x.T, extent=[-0.3, 0.3, 0, 3.0], origin="lower", aspect="auto")
# # plt.colorbar()
# # plt.show()
# plt.imshow(A_y.T, extent=[-0.3, 0.3, 0, 3.0], origin="lower", aspect="auto")
# plt.xlabel(r"$J_\pm/J_{yy}$")
# plt.ylabel(r"$h/J_{yy}$")
# plt.title(r"$\partial M/\partial J_{\pm}$")
# plt.colorbar()
# plt.show()

#
# C = plt.imshow(A.T, origin='lower', aspect='equal', extent=[-0.3,0.1,0,0.5])
# plt.xlabel(r'$J_\pm/J_{y}$')
# plt.ylabel(r'$h/J_{y}$')
# # plt.show()
# plt.savefig('phase_110_kappa=2_Jpmpm=0.2.pdf')
# plt.clf()

# conclude_XYZ_finite_field("/scratch/zhouzb79/phase_110_mag_phase_Jpmpm=-0.2", -0.3,0.1,0,0.5)
# conclude_XYZ_finite_field("/scratch/zhouzb79/phase_001_mag_phase_Jpmpm=-0.2",-0.3,0.1,0,0.5)
# conclude_XYZ_finite_field("/scratch/zhouzb79/phase_111_kappa=2_Jpmpm=-0.2",-0.3,0.1,0,0.5)


# N=1
# Jpm = 0.03
# Jpmpm = np.linspace(0.0, 0.47, N)
# E = np.zeros(N)
# cond = np.zeros(N)
# phi = np.zeros(N)
# order = np.zeros(N, dtype= np.complex128)
#
# # Jxx, Jyy, Jzz = -2 * (Jpm ), 1., 2 * ( - Jpm)
# # ref = pycon.piFluxSolver(Jxx, Jyy, Jzz, flux=np.zeros(4), h=0.0, n=h001, simplified=False)
# # ref.solvemeanfield()
# # ref_energy = ref.GS()
#
# for i in range(N):
#     Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm[i]),  1.,        2*(Jpmpm[i]-Jpm)
#     # Jxx, Jyy, Jzz = -0.5,     1,         1
#     # # # # Jxx, Jyy, Jzz = 1,  0.4,         0.2
#     fig, axs = plt.subplots()
#     a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=np.zeros(4), h=0.0, n=h001, simplified=False)
#     # a.solvemeanfield(Fast=True, ref_energy=ref_energy)
#     a.solvemeanfield()
#     E[i] = a.MFE()
#     cond[i] = a.condensed
#     phi[i] = np.linalg.norm(a.rhos)
#     order[i] = a.order()
# plt.plot(Jpmpm, E)
# plt.savefig("MFE_Jpm=-0.1.pdf")
# plt.clf()
# plt.plot(Jpmpm, cond)
# plt.savefig("Condensed_Jpm=-0.1.pdf")
# plt.clf()
# plt.plot(Jpmpm, phi)
# plt.savefig("Condensed_Jpm=-0.1.pdf")
# plt.clf()
# print(order)
# NJpm = 20
# NH = 20
# Jpm = np.linspace(-0.1, 0.1, NJpm)
# h = np.linspace(0.0, 0.5, NH)
# E = np.zeros((NJpm,NH))
# Epi = np.zeros((NJpm,NH))
# E_3 = np.zeros((NJpm,NH))
# E_4 = np.zeros((NJpm,NH))



# C = np.zeros((NJpm,NH))
# Cpi = np.zeros((NJpm,NH))
# C_3 = np.zeros((NJpm,NH))
# C_4 = np.zeros((NJpm,NH))

# phase = np.zeros((NJpm,NH))

# fig, axs = plt.subplots()

# for i in range(NJpm):
#     for j in range(NH):
#         Jxx, Jyy, Jzz = -2*(Jpm[i]),  1.,        2*(-Jpm[i])
#         a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=np.zeros(4)*np.pi, h=h[j], n=h111, simplified=False)
#         a.solvemeanfield()
#         E[i,j] = a.MFE()
#         C[i,j] = a.condensed

#         a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=np.ones(4)*np.pi, h=h[j], n=h111, simplified=False)
#         a.solvemeanfield()
#         Epi[i,j] = a.MFE()
#         Cpi[i,j] = a.condensed

#         a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=FFFluxGen(-np.pi/3), h=h[j], n=h111, simplified=False, FF=True)
#         a.solvemeanfield()
#         E_3[i,j] = a.MFE()
#         C_3[i,j] = a.condensed

#         # a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=FFFluxGen(np.pi/4), h=h[j], n=h111, simplified=False, FF=True)
#         # a.solvemeanfield()
#         # E_4[i,j] = a.MFE()
#         # C_4[i,j] = a.condensed


#         temp = np.array([E[i,j], Epi[i,j], E_3[i,j]])
#         tempC = np.array([C[i,j], Cpi[i,j], C_3[i,j]])
#         print(temp)
#         min = np.argmin(temp)
#         if tempC[min] == 0:
#             phase[i,j] = min
#         else:
#             phase[i,j] = np.nan
#     # a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=FFFluxGen(np.pi/6), h=h[i], n=h111, simplified=False, FF=True)
#     # a.solvemeanfield()
#     # E_6[i] = a.MFE()
# np.savetxt("FFvsPi_energy_0.txt", E)
# np.savetxt("FFvsPi_energy_pi.txt", Epi)
# np.savetxt("FFvsPi_energy_pi3.txt", E_3)
# np.savetxt("FFvsPi_energy_pi4.txt", E_4)


# np.savetxt("FFvsPi_condensed_0.txt", C)
# np.savetxt("FFvsPi_condensed_pi.txt", Cpi)
# np.savetxt("FFvsPi_condensed_pi3.txt", C_3)
# np.savetxt("FFvsPi_condensed_pi4.txt", C_4)

# plt.imshow(phase.T, origin='lower', extent=[-0.1, 0.1, 0, 0.5], aspect='auto')
# plt.savefig("phase_w_FF.pdf")
# plt.clf()


# # a.graph_loweredge(False,axs,'b')
# # a.graph_upperedge(True,axs,'b')
# A = a.MFE()
# AC = a.condensed
# print(a.rhos, a.chi, a.xi, a.magnetization(), a.gap(), a.MFE())
fig, axs = plt.subplots()
#2420158631264392

# Jpm = -0.1
# Jpmpm = 0.2
# Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm),  1.,        2*(Jpmpm-Jpm)
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, 0, flux=np.ones(4)*np.pi, h=0.1, n=h110, simplified=False)
# a.solvemeanfield()
# # a.graph_loweredge(False,axs,'b')
# # a.graph_upperedge(True,axs,'b')
# A = a.MFE()
# AC = a.condensed
# print(a.chi, a.xi,a.magnetization(), a.gap(), a.MFE())
# a.graph(axs)
# plt.show()

# SSSF_BZ(50, 0.08, 0.08, 1, 0, h110, np.zeros(4),15, "SSSF_BZ", 0)
# SSSF_q_omega_beta(0, 200, 50, 0.08, 0.08, 1, 0, h110, np.zeros(4), 25, "SSSF_0_flux_T=0", "hhl" )
# SSSF_q_omega_beta(0.2, 200, 50, 0.08, 0.08, 1, 0, h110, np.zeros(4), 25, "SSSF_0_flux_T=0.2", "hhl" )
# SSSF_q_omega_beta(0.4, 200, 50, 0.08, 0.08, 1, 0, h110, np.zeros(4), 25, "SSSF_0_flux_T=0.4", "hhl" )
# SSSF_q_omega_beta(0.6, 200, 50, 0.08, 0.08, 1, 0, h110, np.zeros(4), 25, "SSSF_0_flux_T=0.6", "hhl" )
# SSSF_q_omega_beta(0.8, 200, 50, 0.08, 0.08, 1, 0, h110, np.zeros(4), 25, "SSSF_0_flux_T=0.8", "hhl" )
# SSSF_q_omega_beta(1.0, 200, 50, 0.08, 0.08, 1, 0, h110, np.zeros(4), 25, "SSSF_0_flux_T=1.0", "hhl" )

# A = np.loadtxt("spec_heat.txt", unpack=True)
# # plt.imshow(A, origin="lower", aspect="auto")
# # plt.show()
# A = np.flip(A, axis=1)
# A = A[:,10:]
# A[1] = A[1] * 8.6173303e-2
# Cv_integrand = A[1]/A[0]
# S = np.zeros(len(A[1])-1)
# for i in range(1,len(A[1])):
#     S[i-1] = -np.trapezoid(Cv_integrand[i:], A[0][i:]) + np.log(2)
# plt.plot(A[0,:-1],S, A[0], A[1])
# plt.errorbar(A[0], A[1], A[2])
# plt.xscale('log')
# plt.xlabel(r'$T/|J_{yy}|$')
# plt.legend(['entropy', 'specific heat'])
# plt.show()
# dir = "Classical_Phase_Diagram"
# directory = os.fsencode(dir)
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".txt"):
#         print(filename)
#         A = np.loadtxt(dir+"/"+filename)
#         plt.imshow(A.T, extent=[-0.3,0.3,0,15], aspect='auto', origin='lower')
#         plt.xlabel(r'$J_\pm/J_{yy}$')
#         plt.ylabel(r'$h/J_{yy}$')
#         plt.colorbar()
#         plt.savefig(dir+"/"+filename[:-4]+'.pdf')
#         plt.clf()
#         if filename.endswith("_global_mag.txt"):
#             A_x, A_y = np.gradient(A)
#             plt.imshow(A_x.T, extent=[-0.3,0.3,0,15], aspect='auto', origin='lower')
#             plt.xlabel(r'$J_\pm/J_{yy}$')
#             plt.ylabel(r'$h/J_{yy}$')
#             plt.savefig(dir+"/"+filename[:-4]+'_dH.pdf')
#             plt.clf()
#             plt.imshow(A_y.T, extent=[-0.3,0.3,0,15], aspect='auto', origin='lower')
#             plt.xlabel(r'$J_\pm/J_{yy}$')
#             plt.ylabel(r'$h/J_{yy}$')
#             plt.savefig(dir+"/"+filename[:-4]+'_dJpm.pdf')
#             plt.clf()
# Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm),  1.,        2*(Jpmpm-Jpm)
# Jxx, Jyy, Jzz = -0.06,     1,         -0.06
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=np.ones(4) * np.pi, h=0.2, n=h111,simplified=False)
# a.solvemeanfield()
# a.graph(axs)
# plt.show()
# print(a.magnetization())

# Jxx, Jyy, Jzz = -0.05,     1,         -0.05
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=np.zeros(4) * np.pi, h=0.2, n=h111,simplified=True)
# a.solvemeanfield()
# # a.graph(axs)
# print(a.magnetization())
# print(a.chi, a.xi, a.magnetization(),a.gap(), a.MFE())
# # #
# # Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm),  1.,        2*(Jpmpm-Jpm)
# # fig, axs = plt.subplots()
# a = pycon.piFluxSolver(0.234338, 1.0, 0.379298, theta=-0.334922, flux=np.zeros(4) * np.pi, h=0.1, n=h110,simplified=False)
# a.solvemeanfield()
# a.graph(axs)
# B = a.MFE()
# BC = a.condensed
# print(B, BC)
# plt.show()
# print(a.chi, a.xi, a.magnetization(),a.gap(), a.MFE())

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
# Jpm = -0.1
# Jpmpm = -0.2
#
# fig, axs = plt.subplots(nrows=2, figsize=(10, 8), layout="constrained", sharex=True)
# axs[0].text(.01, .99, r"$(\mathrm{a})$", ha='left', va='top', transform=axs[0].transAxes,
#                zorder=10)
# axs[1].text(.01, .99, r"$(\mathrm{b})$", ha='left', va='top', transform=axs[1].transAxes,
#                zorder=10)
# a = pycon.piFluxSolver(-2*Jpmpm-2*Jpm, 1, 2*Jpmpm-2*Jpm, flux=np.pi*np.ones(4), h=0.15, n=h110, simplified=True)
# a.solvemeanfield()
# # a.graph(axs[0])
#
# emin, emax = a.graph_loweredge(False, axs[0]), a.graph_upperedge(False, axs[0])
# d = np.loadtxt("../XYZ_project/Szzglobal_DSSF.txt")
# emin, emax = np.min(emin) * 0.95, np.max(emax) * 1.02
# # d = DSSFparse(emin, emax, d)
# c = axs[0].imshow(d.T / np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax],
#                      aspect='auto', cmap='gnuplot2')
# fig.colorbar(c, ax=axs[0])
#
# Jpm = -0.06
# Jpmpm = -0.2
#
# a = pycon.piFluxSolver(-2*Jpmpm-2*Jpm, 1, 2*Jpmpm-2*Jpm, flux=np.pi*np.ones(4), h=0.2, n=h111, simplified=True)
# a.solvemeanfield()
# # a.graph(axs[1])
#
# emin, emax = a.graph_loweredge(False, axs[1]), a.graph_upperedge(False, axs[1])
# d = np.loadtxt("../XYZ_project/Szzglobal_DSSF_111.txt")
# emin, emax = np.min(emin) * 0.95, np.max(emax) * 1.02
# # d = DSSFparse(emin, emax, d)
# c = axs[1].imshow(d.T / np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax],
#                      aspect='auto', cmap='gnuplot2')
# fig.colorbar(c, ax=axs[1])
#
# plt.savefig("DSSF.pdf")
# plt.clf()

#region graphing functions for XYZ project
# mpl.rcParams.update({'font.size': 40})

# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), layout="constrained")
# ax00 = axs[0]
# ax01 = axs[1]
# ax02 = axs[2]
# ax00.text(.01, .99, r"$(\mathrm{a})$", ha='left', va='top', transform=ax00.transAxes,
#                zorder=10,color='black')
# ax01.text(.01, .99, r"$(\mathrm{b})$", ha='left', va='top', transform=ax01.transAxes,
#                zorder=10,color='w')
# ax02.text(.01, .99, r"$(\mathrm{c})$", ha='left', va='top', transform=ax02.transAxes,
#                zorder=10,color='black')

# A = np.loadtxt("../XYZ_project/SSSF/SSSF_CHO_h110_0.1/Szzglobal.txt")
# c = SSSFgenhelper(A /np.max(A), "110", ax00, fig, False)
# fig.colorbar(c, ax=ax00)
# ax00.set_xlabel(r"$(h,h,0)$")
# ax00.set_ylabel(r"$(0,0,l)$")
# # ax00.set_title(r"$J_{\pm\pm}=0$")

# A = np.loadtxt("../XYZ_project/SSSF/SSSF_CHO_h110_0.1_0_flux/Szzglobal.txt")
# c = SSSFgenhelper(A /np.max(A), "110", ax01, fig, False)
# fig.colorbar(c, ax=ax01)
# ax01.set_xlabel(r"$(h,h,0)$")
# # ax01.set_ylabel(r"$(0,0,l)$")
# # ax00.set_title(r"$J_{\pm\pm}=0$")

# A = np.loadtxt("../XYZ_project/SSSF/SSSF_XYZ_110_Jpm_-0.1_Jpmpm=-0.2_h_0.15/Szzglobal.txt")
# c = SSSFgenhelper(A /np.max(A), "110", ax02, fig, False)
# fig.colorbar(c, ax=ax02)
# ax02.set_xlabel(r"$(h,h,0)$")
# # ax01.set_ylabel(r"$(0,0,l)$")
# # ax02.set_title(r"$J_{\pm\pm}=-0.2$")
# plt.savefig("SSSF_global_same.pdf")
# plt.clf()

fig = plt.figure(figsize=(22, 8), constrained_layout=True)
spec = gridspec.GridSpec(ncols=6, nrows=3, figure=fig, height_ratios=[0.05,1,1])
spec.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)


ax00 = fig.add_subplot(spec[1, 0])
ax01 = fig.add_subplot(spec[1, 1])
ax02 = fig.add_subplot(spec[1, 2])
ax03 = fig.add_subplot(spec[1, 3])
ax04 = fig.add_subplot(spec[1, 4])
ax05 = fig.add_subplot(spec[1, 5])

ax1 = fig.add_subplot(spec[2, 0:3])
ax2 = fig.add_subplot(spec[2, 3:])

colobar0 = fig.add_subplot(spec[0, 0])
colobar1 = fig.add_subplot(spec[0, 1])
colobar2 = fig.add_subplot(spec[0, 2])
colobar3 = fig.add_subplot(spec[0, 3])
colobar4 = fig.add_subplot(spec[0, 4])
colobar5 = fig.add_subplot(spec[0, 5])

ax00.text(.01, .99, r"$(\mathrm{1.a})$", ha='left', va='top', transform=ax00.transAxes,
               zorder=10,color='w')
ax01.text(.01, .99, r"$(\mathrm{1.b})$", ha='left', va='top', transform=ax01.transAxes,
               zorder=10,color='w')
ax02.text(.01, .99, r"$(\mathrm{1.c})$", ha='left', va='top', transform=ax02.transAxes,
               zorder=10,color='w')
ax03.text(.01, .99, r"$(\mathrm{2.a})$", ha='left', va='top', transform=ax03.transAxes,
               zorder=10,color='w')
ax04.text(.01, .99, r"$(\mathrm{2.b})$", ha='left', va='top', transform=ax04.transAxes,
               zorder=10,color='w')
ax05.text(.01, .99, r"$(\mathrm{2.c})$", ha='left', va='top', transform=ax05.transAxes,
               zorder=10,color='w')

ax1.text(.01, .99, r"$(\mathrm{1.d})$", ha='left', va='top', transform=ax1.transAxes,
               zorder=10,color='w')
ax2.text(.01, .99, r"$(\mathrm{2.d})$", ha='left', va='top', transform=ax2.transAxes,
               zorder=10,color='w')

A = np.loadtxt("../XYZ_project/SSSF/SSSF_CHO_h110_0.1/Stotal_global.txt")
c = SSSFgenhelper(A /np.max(A), "110", ax00, fig, True)
ax00.set_xlabel(r"$(h,-h,0)$")
ax00.set_ylabel(r"$(0,0,l)$")
colobar0.set_title(r"$\mathcal{S}^{zz}$")
# fig.colorbar(c, ax=ax00)
Colorbar(mappable=c, ax=colobar0, orientation = 'horizontal', ticklocation = 'top')

A = np.loadtxt("../XYZ_project/SSSF/SSSF_CHO_h110_0.1/Stotal_NSF.txt")
SSSFgenhelper(A /np.max(A), "110", ax01, fig, False)
ax01.set_xlabel(r"$(h,-h,0)$")
# ax01.set_ylabel(r"$(0,0,l)$")
ax01.set_yticklabels([])
# ax01.set_xticklabels([])

colobar1.set_title(r"$\mathcal{S}_{NSF}^{zz}$, $\pi$-$flux$")
Colorbar(mappable=c, ax=colobar1, orientation = 'horizontal', ticklocation = 'top')

A = np.loadtxt("../XYZ_project/SSSF/SSSF_CHO_h110_0.1_0_flux/Stotal_NSF.txt")
c = SSSFgenhelper(A /np.max(A), "110", ax02, fig, False)
ax02.set_xlabel(r"$(h,-h,0)$")
ax02.set_yticklabels([])
# ax02.set_xticklabels([])
# ax02.set_ylabel(r"$(0,0,l)$")
colobar2.set_title(r"$\mathcal{S}_{NSF}^{zz}$, $0$-$flux$")
Colorbar(mappable=c, ax=colobar2, orientation = 'horizontal', ticklocation = 'top')

d = np.loadtxt("../XYZ_project/DSSF/DSSF_CHO_h110_0.4/S_total_global.txt")
a = pycon.piFluxSolver(0.234338, 1.0, 0.379298,theta=-0.334922,flux=np.ones(4)*np.pi,h=0.1,n=h110, simplified=False)
a.solvemeanfield()
emin, emax = 0.95*np.min(a.graph_loweredge(False, ax1)), 1.02*np.max(a.graph_upperedge(False, ax1))
# emin, emax = 0.29835552586608627, 1.8698587798053417
print(emin, emax)
ax1.set_ylabel(r"$\omega/J_{\parallel}$")
c = ax1.imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax], aspect='auto', cmap='gnuplot2')
# fig.colorbar(c, ax=ax1)

A = np.loadtxt("../XYZ_project/SSSF/SSSF_XYZ_110_Jpm_-0.1_Jpmpm=-0.2_h_0.15/Szzglobal.txt")
c = SSSFgenhelper(A /np.max(A), "110", ax03, fig, True)
ax03.set_xlabel(r"$(h,-h,0)$")
# ax03.set_ylabel(r"$(0,0,l)$")
ax03.set_yticklabels([])
# ax03.set_xticklabels([])
colobar3.set_title(r"$\mathcal{S}^{zz}$")
# fig.colorbar(c, ax=ax03)
Colorbar(mappable=c, ax=colobar3, orientation = 'horizontal', ticklocation = 'top')

A = np.loadtxt("../XYZ_project/SSSF/SSSF_XYZ_110_Jpm_-0.1_Jpmpm=-0.2_h_0.15/SzzNSF.txt")
SSSFgenhelper(A /np.max(A), "110", ax04, fig, False)
ax04.set_xlabel(r"$(h,-h,0)$")
# ax01.set_ylabel(r"$(0,0,l)$")
ax04.set_yticklabels([])
# ax04.set_xticklabels([])
colobar4.set_title(r"$\mathcal{S}_{NSF}^{zz}$, $J_{\pm\pm}=-0.2J_\parallel$")
Colorbar(mappable=c, ax=colobar4, orientation = 'horizontal', ticklocation = 'top')

A = np.loadtxt("../XYZ_project/SSSF/SSSF_XYZ_110_Jpm_-0.1_Jpmpm=0.0_h_0.15/SzzNSF.txt")
c = SSSFgenhelper(A /np.max(A), "110", ax05, fig, False)
ax05.set_xlabel(r"$(h,-h,0)$")
ax05.set_yticklabels([])
# ax05.set_xticklabels([])

# ax02.set_ylabel(r"$(0,0,l)$")
colobar5.set_title(r"$\mathcal{S}_{NSF}^{zz}$, $J_{\pm\pm}=0$")
Colorbar(mappable=c, ax=colobar5, orientation = 'horizontal', ticklocation = 'top')

d = np.loadtxt("../XYZ_project/DSSF/DSSF_XYZ_110_Jpm_-0.1_Jpmpm=-0.2_h_0.15/Szzglobal.txt")
Jpm=-0.1
Jpmpm=-0.2
a = pycon.piFluxSolver(0.6, 1.0, -0.2,theta=0,flux=np.ones(4)*np.pi,h=0.15,n=h110, simplified=False)
a.solvemeanfield()
emin, emax = 0.95*np.min(a.graph_loweredge(False, ax2)), 1.02*np.max(a.graph_upperedge(False, ax2))
# emin, emax = 0.29835552586608627, 1.8698587798053417
print(emin, emax)
c = ax2.imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax], aspect='auto', cmap='gnuplot2')
fig.colorbar(c, ax=ax2)

plt.savefig("CHO_SSF_all.pdf")


# axs[1].set_ylabel(r"$(0,0,l)$")
# A = np.loadtxt("../XYZ_project/SSSF/SSSF_CHO_h110_0.1/SzzNSF.txt")
# SSSFgenhelper(A /np.max(A), "110", axs[1,0], fig)
# A = np.loadtxt("../XYZ_project/SSSF/SSSF_CHO_h110_0.1_0_flux/SzzNSF.txt")
# SSSFgenhelper(A /np.max(A), "110", axs[1,1], fig)

# fig, ax1 = plt.subplots(figsize=(12,5))
# d = np.loadtxt("../XYZ_project/DSSF/DSSF_XYZ_110_Jpm_-0.1_Jpmpm=-0.2_h_0.15/Szzglobal.txt")
# Jpm = -0.1
# Jpmpm = -0.2
# a = pycon.piFluxSolver(-2*Jpm-2*Jpmpm, 1.0, -2*Jpm+2*Jpmpm,theta=0,flux=np.ones(4)*np.pi,h=0.15,n=h110, simplified=False)
# a.solvemeanfield()
# emin, emax = np.min(a.graph_loweredge(False, ax1)), np.max(a.graph_upperedge(False, ax1))
# # emin, emax = 0.29835552586608627, 1.8698587798053417
# print(emin, emax)
# c = ax1.imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax], aspect='auto', cmap='gnuplot2')
# fig.colorbar(c, ax=ax1)

# plt.savefig("CZO_DSSF.pdf")
#endregion


# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), layout="constrained", sharex=True)
# mpl.rcParams.update({'font.size': 25})
# axs[0, 0].text(.01, .99, r"$(\mathrm{a})$", ha='left', va='top', transform=axs[0, 0].transAxes,
#                zorder=10)
# axs[0, 1].text(.01, .99, r"$(\mathrm{b})$", ha='left', va='top', transform=axs[0, 1].transAxes,
#                zorder=10)
# axs[1, 0].text(.01, .95, r"$(\mathrm{c})$", ha='left', va='top', transform=axs[1, 0].transAxes, zorder=10)
# axs[1, 1].text(.01, .95, r"$(\mathrm{d})$", ha='left', va='top', transform=axs[1, 1].transAxes, zorder=10)
# A = np.loadtxt("../XYZ_project/magneto_110_111_CSO_O.txt")
#
# A = np.loadtxt("../XYZ_project/magneto_110_111_CSO_D.txt")
# A = np.loadtxt("../XYZ_project/magneto_110_111_CHO_O.txt")
# A = np.loadtxt("../XYZ_project/magneto_110_111_CHO_D.txt")
# plt.savefig("SSSF.pdf")

# Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm),  1.,        2*(Jpmpm-Jpm)
fig, axs = plt.subplots()
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, flux=np.zeros(4) * np.pi, h=0.2, n=h111)
# a.solvemeanfield()
# # a.graph(axs)
# print(a.magnetization())
# C = a.MFE()
# CC = a.condensed
#
#
# # Jpmpm=0.2
# # # Jxx, Jyy, Jzz = -2*(Jpm+Jpmpm),  1.,        2*(Jpmpm-Jpm)
# fig, axs = plt.subplots()
# a = pycon.piFluxSolver(Jxx,Jyy, Jzz, 1, flux=np.zeros(4) * np.pi, h=0.2, n=h111, simplified=True)
# a.solvemeanfield()
# # a.graph(axs)
# D = a.MFE()
# DC = a.condensed
#
# # a.graph(axs)
# print(a.magnetization())

# # plt.show()
# print(A, B, C, D)
# print(AC, BC, CC, DC)
# print(C,D,E)
# print(CC,DC,EC)
# print(A, C)
# conclude_XYZ_0_field("../Data/phase_diagrams/phase_XYZ_0_field")
# conclude_XYZ_0_field("Misc/New folder/phase_XYZ_0_field")
# conclude_XYZ_0_field("../../Data_Archive/pyrochlore_gmft/phase_XYZ_not_symm/phase_XYZ_0_field", -1, 1)
# conclude_XYZ_0_field("../../Data_Archive/Files/phase_XYZ_0_field",-0.8, 1.0)

# conclude_XYZ_0_field_job_array("/scratch/zhouzb79/Files/pyrochlore_XYZ_0_field_no_condensate")
# conclude_XYZ_finite_field_job_array("/scratch/zhouzb79/pyrochlore_mag_phase_001_Jpmpm=-0.2")
conclude_XYZ_finite_field_job_array("/scratch/zhouzb79/pyrochlore_mag_phase_001_Jpmpm=0.03624000000000001")
# conclude_XYZ_finite_field_job_array("/scratch/zhouzb79/pyrochlore_mag_phase_110_Jpmpm=-0.2")

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
# d3 = np.loadtxt("../Data/Fi gvnal_DSSF_pedantic/Jpm=-0.289_pi/h_110/h=0.09999999999999999/Szz/13.txt")
#
# d = d0+d1+d2+d3
# c = axs.imshow(d.T/np.max(d))
# fig.colorbar(c)
# plt.show()
# mpl.rcParams.update({'font.size': 25})
# fig, axs = plt.subplots()
# Jpm=0.0
# flux = np.zeros(4)
# # flux = np.ones(4)*np.pi d
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

# TwoSpinonDOS_111(50, 30, "two_spinon_DOS_111_Jpm=-0.1.pdf")
# plt.clf()
# # #s
# TwoSpinonDOS_111(50, 30, "two_spinon_DOS_001_Jpm=-0.1_Jpmpm=0.2.pdf")
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
