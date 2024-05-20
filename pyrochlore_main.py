import os
import matplotlib as mpl
import matplotlib.font_manager as font_manager

mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
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
                temp = foldname+'/'+files
                d = np.loadtxt(temp)
                d = d/np.max(d)
                kline = np.concatenate((graphGammaX, graphXW, graphWK, graphKGamma, graphGammaL, graphLU, graphUW1, graphW1X1, graphX1Gamma))
                e = np.linspace(0,1,len(d))
                X, Y = np.meshgrid(kline, e)
                DSSFgraph(d.T, temp[:-4], X, Y)

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


def DSSFgraphGen(h0, hmid, mid, hpi, n, filename):
    mpl.rcParams.update({'font.size': 22})
    plt.margins(x=0.04,y=0.04)

    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(22, 10),layout="constrained", sharex=True)

    axs[0,0].text(.01, .99, r"$a)$", ha='left', va='top', transform=axs[0,0].transAxes, zorder=10)
    axs[0,1].text(.01, .99, r"$b)$", ha='left', va='top', transform=axs[0,1].transAxes, zorder=10)
    axs[0,2].text(.01, .99, r"$c)$", ha='left', va='top', transform=axs[0,2].transAxes, zorder=10)
    axs[1,0].text(.01, .99, r"$d)$", ha='left', va='top', transform=axs[1,0].transAxes, color='w', zorder=10)
    axs[1,1].text(.01, .99, r"$e)$", ha='left', va='top', transform=axs[1,1].transAxes, color='w', zorder=10)
    axs[1,2].text(.01, .99, r"$f)$", ha='left', va='top', transform=axs[1,2].transAxes, color='w', zorder=10)

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

    a = pycon.piFluxSolver(-0.04,-0.04,1,flux=np.zeros(4),h=h0,n=n)
    a.solvemeanfield()
    a.graph(axs[0,0])

    emin, emax = a.graph_loweredge(False, axs[1,0]), a.graph_upperedge(False, axs[1,0])
    d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=0.02_0/h_"+dirString+"/h="+str(h0)+"/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    c = axs[1,0].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax], aspect='auto')
    fig.colorbar(c, ax=axs[1,0])

    a = pycon.piFluxSolver(0.06,0.06,1,flux=mid,h=hmid,n=n)
    a.solvemeanfield()
    a.graph(axs[0,1])

    emin, emax = a.graph_loweredge(False, axs[1,1]), a.graph_upperedge(False, axs[1,1])
    d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.03_"+fluxString+"/h_"+dirString+"/h="+str(hmid)+"/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    c = axs[1,1].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax], aspect='auto')
    fig.colorbar(c, ax=axs[1,1])


    a = pycon.piFluxSolver(0.289*2,0.289*2,1,flux=np.ones(4)*np.pi,h=hpi,n=n)
    a.solvemeanfield()
    a.graph(axs[0,2])

    emin, emax = a.graph_loweredge(False, axs[1,2]), a.graph_upperedge(False, axs[1,2])
    d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_"+dirString+"/h="+str(hpi)+"/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    c = axs[1,2].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax], aspect='auto')
    fig.colorbar(c, ax=axs[1,2])


    plt.savefig(filename)
def DSSFgraphGen_0(n, filename):
    mpl.rcParams.update({'font.size': 25})
    plt.margins(x=0.04,y=0.04)

    fig, axs = plt.subplots(nrows=2,ncols=2, figsize=(16, 8),layout="constrained", sharex=True)

    axs[0,0].text(.01, .99, r"$a)$", ha='left', va='top', transform=axs[0,0].transAxes, zorder=10)
    axs[0,1].text(.01, .99, r"$b)$", ha='left', va='top', transform=axs[0,1].transAxes, zorder=10)
    axs[1,0].text(.01, .99, r"$d)$", ha='left', va='top', transform=axs[1,0].transAxes, color='w', zorder=10)
    axs[1,1].text(.01, .99, r"$e)$", ha='left', va='top', transform=axs[1,1].transAxes, color='w', zorder=10)
    axs[0,0].set_ylabel(r'$\omega/J_{yy}$')
    axs[1,0].set_ylabel(r'$\omega/J_{yy}$')
    dirString = ""
    if (n==h110).all():
        dirString = "110"
    elif (n==h111).all():
        dirString = "111"
    else:
        dirString = "001"

    a = pycon.piFluxSolver(-0.04,-0.04,1,flux=np.zeros(4),h=0,n=n)
    a.solvemeanfield()
    a.graph(axs[0,0])

    emin, emax = a.graph_loweredge(False, axs[1,0]), a.graph_upperedge(False, axs[1,0])
    d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=0.02_0/h_"+dirString+"/h=0.0/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    c = axs[1,0].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax], aspect='auto')
    fig.colorbar(c, ax=axs[1,0])


    a = pycon.piFluxSolver(0.289*2,0.289*2,1,flux=np.ones(4)*np.pi,h=0,n=n)
    a.solvemeanfield()
    a.graph(axs[0,1])

    emin, emax = a.graph_loweredge(False, axs[1,1]), a.graph_upperedge(False, axs[1,1])
    d = np.loadtxt("../Data/Final_DSSF_pedantic/Jpm=-0.289_pi/h_"+dirString+"/h=0.0/Szzglobal.txt")
    emin, emax = np.min(emin) * 0.95, np.max(emax)*1.02
    c = axs[1,1].imshow(d.T/np.max(d), interpolation="lanczos", origin='lower', extent=[0, gGamma3, emin, emax], aspect='auto')
    fig.colorbar(c, ax=axs[1,1])

    plt.savefig(filename)

def SSSFgraphGen(h0, hmid, fluxmid, hpi, n, tograph, filename, colors=np.array(['w','w','w','w','w','w','w','w','w','w'])):
    mpl.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(nrows=3,ncols=3, figsize=(16, 14), layout="tight", sharex=True, sharey=True)

    if len(hpi) == 2:
        axs[-1, -1].axis('off')

    axs[0,0].set_xlabel(r'$J_\pm/J_{yy}=0.02$', labelpad=10)
    axs[0,0].xaxis.set_label_position('top')
    axs[0,1].set_xlabel(r'$J_\pm/J_{yy}=-0.03$', labelpad=10)
    axs[0,1].xaxis.set_label_position('top')
    axs[0,2].set_xlabel(r'$J_\pm/J_{yy}=-0.3$', labelpad=10)
    axs[0,2].xaxis.set_label_position('top')

    fig.text(1, 0.85, r'$h/J_{yy}=' + str(h0[0]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.52, r'$h/J_{yy}=' + str(h0[1]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.2, r'$h/J_{yy}=' + str(h0[2]) + '$', ha='right', va='center', rotation=270)

    dirString = ""
    if (n==h110).all():
        dirString = "110"
        axs[0, 0].set_ylabel(r'$(0,0,l)$')
        axs[1, 0].set_ylabel(r'$(0,0,l)$')
        axs[2, 0].set_ylabel(r'$(0,0,l)$')
        axs[2, 0].set_xlabel(r'$(h,-h,0)$')
        axs[2, 1].set_xlabel(r'$(h,-h,0)$')
        axs[1, 2].set_xlabel(r'$(h,-h,0)$')

    elif (n==h111).all():
        dirString = "111"
        axs[0, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[1, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[2, 0].set_ylabel(r'$(k,k,-2k)$')
        axs[2, 0].set_xlabel(r'$(h,-h,0)$')
        axs[2, 1].set_xlabel(r'$(h,-h,0)$')
        axs[2, 2].set_xlabel(r'$(h,-h,0)$')
    else:
        dirString = "001"
        axs[0, 0].set_ylabel(r'$(0,k,0)$')
        axs[1, 0].set_ylabel(r'$(0,k,0)$')
        axs[2, 0].set_ylabel(r'$(0,k,0)$')
        axs[2, 0].set_xlabel(r'$(h,0,0)$')
        axs[2, 1].set_xlabel(r'$(h,0,0)$')
        axs[1, 2].set_xlabel(r'$(h,0,0)$')


    for i in range(len(h0)):
        d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=0.02_0/h_"+dirString+"/h="+str(h0[i])+"/"+tograph)
        SSSFgenhelper(d/np.max(d), dirString, axs[i,0], fig)

    for i in range(len(hmid)):
        d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.03_"+ fluxmid[i] +"/h_"+dirString+"/h="+str(hmid[i])+"/"+tograph)
        SSSFgenhelper(d/np.max(d), dirString, axs[i,1], fig)

    for i in range(len(hpi)):
        d = np.loadtxt("../Data/Final_SSSF_pedantic/Jpm=-0.289_pi/h_"+dirString+"/h="+str(hpi[i])+"/"+tograph)
        SSSFgenhelper(d/np.max(d), dirString, axs[i,2], fig)

    axs[0,0].text(.01, .99, r"$a)$", ha='left', va='top', transform=axs[0,0].transAxes, color=colors[0])
    axs[0,1].text(.01, .99, r"$b)$", ha='left', va='top', transform=axs[0,1].transAxes, color=colors[1])
    axs[0,2].text(.01, .99, r"$c)$", ha='left', va='top', transform=axs[0,2].transAxes, color=colors[2])
    axs[1,0].text(.01, .99, r"$d)$", ha='left', va='top', transform=axs[1,0].transAxes, color=colors[3])
    axs[1,1].text(.01, .99, r"$e)$", ha='left', va='top', transform=axs[1,1].transAxes, color=colors[4])
    axs[1,2].text(.01, .99, r"$f)$", ha='left', va='top', transform=axs[1,2].transAxes, color=colors[5])
    axs[2,0].text(.01, .99, r"$g)$", ha='left', va='top', transform=axs[2,0].transAxes, color=colors[6])
    axs[2,1].text(.01, .99, r"$h)$", ha='left', va='top', transform=axs[2,1].transAxes, color=colors[7])
    if not len(hpi) == 2:
        axs[2,2].text(.01, .99, r"$i)$", ha='left', va='top', transform=axs[2,2].transAxes, color=colors[8])
    # plt.show()
    plt.savefig(filename)

def sublatticeSSSFgraphGen(n, tograph, h, JP, flux, filename):
    mpl.rcParams.update({'font.size': 30})
    fig, axs = plt.subplots(nrows=len(h)*len(JP),ncols=len(tograph), figsize=(28, 35), layout="tight", sharex=True, sharey=True)
    dirString = ""
    if (n==h110).all():
        dirString = "110"
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

    fig.text(1, 0.91, r'$h/J_{yy}=' + str(h[0]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.74, r'$h/J_{yy}=' + str(h[1]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.58, r'$h/J_{yy}=' + str(h[2]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.42, r'$h/J_{yy}=' + str(h[0]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.26, r'$h/J_{yy}=' + str(h[1]) + '$', ha='right', va='center', rotation=270)
    fig.text(1, 0.1, r'$h/J_{yy}=' + str(h[2]) + '$', ha='right', va='center', rotation=270)
    for i in range(len(tograph)):
        globalS = ""
        if "global" in tograph[i]:
            globalS ='global,'
        temp = tograph[i].split("/")

        if (n==h111).all():
            axs[0, i].set_xlabel(
                r'$\mathcal{S}^{zz}_{' + globalS + '\mathrm{' + temp[-1][0:-4] + '}}\quad J_\pm=' + str(JP[0]) + '$ ',
                labelpad=10)
            axs[0, i].xaxis.set_label_position('top')
            axs[len(h), i].set_xlabel(
                r'$\mathcal{S}^{zz}_{' + globalS + '\mathrm{' + temp[-1][0:-4] + '}}\quad J_\pm=' + str(JP[1]) + '$',
                labelpad=10)
            axs[len(h), i].xaxis.set_label_position('top')
        else:
            axs[0, i].set_xlabel(r'$\mathcal{S}^{zz}_{'+globalS+', '+temp[-1][0:-4]+'}\quad J_\pm='+str(JP[0])+'$ ', labelpad=10)
            axs[0, i].xaxis.set_label_position('top')
            axs[len(h), i].set_xlabel(r'$\mathcal{S}^{zz}_{'+globalS+', '+temp[-1][0:-4]+'}\quad J_\pm='+str(JP[1])+'$', labelpad=10)
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
                if (np.mean(d[-10:,0:10])-np.min(d))/(np.max(d)-np.min(d))>0.5:
                    color = 'black'
                else:
                    color = 'white'

                SSSFgenhelper(d /dmax, dirString, axs[i+j*len(h),k], fig)
                axs[i+j*len(h),k].text(.01, .99, r"$"+str(j+1)+chr(97+i*len(tograph)+k)+")$", ha='left', va='top', transform=axs[i+j*len(h),k].transAxes, color=color)
    plt.savefig(filename)

def phaseRegraphHelp(d, ax, fig, cb=False):
    c = ax.imshow(d.T, interpolation="nearest", origin='lower', extent=[-0.5, 0.1, 0, 1], aspect='auto')
    if cb:
        cb = fig.colorbar(c, ax=ax)
def phaseExGraph(filename):
    mpl.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(16, 8),layout="constrained")

    axs[0,0].set_xlabel(r'$h\parallel(110)$', labelpad=10)
    axs[0,0].xaxis.set_label_position('top')
    axs[0,1].set_xlabel(r'$h\parallel(111)$', labelpad=10)
    axs[0,1].xaxis.set_label_position('top')
    axs[0,2].set_xlabel(r'$h\parallel(001)$', labelpad=10)
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
    plt.savefig(filename)
# phaseExGraph("phase_diagrams_exb.pdf")
# mpl.rcParams.update({'font.size': 30})
# fig, ax = plt.subplots()
# #
# d1 = np.loadtxt("../Data/phase_diagrams/phase_110_kappa=2_zoomed_in_mag.txt")
# h = np.linspace(0,1,len(d1.T))
# ax.plot(h, d1[0],h, d1[40],h, d1[80],h, d1[120],h, d1[160])
# ax.set_xlabel(r"$h/J_{yy}$")
# ax.set_ylabel(r"$\langle S^z \rangle$")
# ax.legend([r'$J_\pm=-0.1$',r'$J_\pm=-0.08$',r'$J_\pm=-0.06$',r'$J_\pm=-0.04$',r'$J_\pm=-0.02$'])
# plt.show()
# DSSF_line_pedantic(20, -0.055, -0.055, 1, 0, 0.1, 2, h110, np.zeros(4), 5, "Files/DSSF/Jpm/")

# ffact = contract('ik, jk->ij', np.array([[0,0,1]]), NN)
# ffact = np.exp(1j * ffact)
# zmag = contract('k,ik->i', h110, z)
# A_pi_here = np.array([[0, 0, 0, 0],
#                       [0, np.pi, np.pi, 0]])
# print(ffact, zmag)
# print('\n')
# print(np.exp(1j*A_pi_here))
#
# fig, axs = plt.subplots()
#
# a = pycon.piFluxSolver(0.03*2,1,0.03*2,flux=np.array([np.pi,np.pi,np.pi,np.pi]), h=0.2, n=h001)
# print(a.A_pi_here)
# a.solvemeanfield()
# a.graph(axs)
# plt.show()
#
# fig, axs = plt.subplots()
#
# a = pycon.piFluxSolver(-0.04,1,-0.04,flux=np.array([np.pi,np.pi,np.pi,np.pi]), h=0.3)
# a.solvemeanfield()
# a.graph(axs)
# plt.show()
# DSSFgraphGen_0(h110,"DSSF_0_field.pdf")
# DSSFgraphGen(0.39999999999999997,0.39999999999999997,np.array([0,0,np.pi,np.pi]),0.09999999999999999,h110,"h110_DSSF.pdf")
# DSSFgraphGen(0.2,0.1,np.array([np.pi,np.pi,np.pi,np.pi]),0.05,h001,"h001_DSSF.pdf")
# DSSFgraphGen(0.2,0.2,np.array([np.pi,np.pi,np.pi,np.pi]),0.2,h111,"h111_DSSF.pdf")

# SSSFgraphGen(np.array([0,0.18,0.42]),np.array([0,0.18,0.42]), np.array(["pi","pi","00pp"]),np.array([0,0.18]),h110,"Szzglobal.txt", "h110_SSSF.pdf", np.array(['w','b','b','w','b','w','w','w','w','w']))
# SSSFgraphGen(np.array([0,0.16,0.32]),np.array([0,0.16,0.32]), np.array(["pi","pi","pi"]),np.array([0,0.16,0.32]),h111,"Szzglobal.txt", "h111_SSSF.pdf", np.array(['w','w','w','w','w','w','w','w','w','w']))
# SSSFgraphGen(np.array([0,0.08,0.2]),np.array([0,0.08,0.2]), np.array(["pi","pi","0"]),np.array([0,0.08]),h001,"Szzglobal.txt", "h001_SSSF.pdf", np.array(['b','w','w','b','w','w','w','w','w','w']))
#
# sublatticeSSSFgraphGen(h110, np.array(["Szz/00.txt", "Szz/12.txt", "Szzglobal/03.txt", "Szzglobal/12.txt"]), np.array([0,0.18,0.42]), np.array([0.02,-0.03]), np.array([["0","0","0"],["pi","pi","00pp"]]), "h110_sublattice.pdf")
sublatticeSSSFgraphGen(h111, np.array(["Szz/Kagome.txt", "Szz/Triangular.txt", "Szzglobal/Kagome.txt", "Szz/Kagome-Tri.txt"]), np.array([0,0.16,0.32]), np.array([0.02,-0.289]), np.array([["0","0","0"],["pi","pi","pi"]]), "h111_sublattice.pdf")
# sublatticeSSSFgraphGen(h001, np.array(["Szz/01.txt", "Szz/03.txt", "Szzglobal/01.txt", "Szzglobal/03.txt"]), np.array([0,0.08,0.2]), np.array([0.02,-0.03]), np.array([["0","0","0"],["pi","pi","0"]]), "h001_sublattice.pdf")


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
