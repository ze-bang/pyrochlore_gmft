import matplotlib.pyplot as plt
# import netCDF4
import numpy as np
import warnings
from misc_helper import *
import pyrochlore_gmft as pycon
import pyrochlore_exclusive_boson as pyex
from mpi4py import MPI
import os

def generaldispersion(Jxx, Jyy, Jzz, h, n, kappa, graphres, BZres, flux):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    py0s.graph(False)
    return 0


def graphPhase(filename):
    phases = np.loadtxt(filename, delimiter=' ').T

    tempc = np.copy(phases[:phases.shape[0]-1, :])
    tempc = np.flip(tempc, axis=0)

    for i in range(tempc.shape[0]):
        for j in range(tempc.shape[1]):
            if (tempc[i][j] == 3) or (tempc[i][j] == -3):
                pass
            else:
                tempc[i][j] = -tempc[i][j]

    # print(phases)
    # print(tempc)

    bigphase = np.concatenate((phases, tempc), axis=0)


    JP = np.linspace(-0.5, 0.1, phases.shape[1])
    kappa = np.linspace(-1, 0, phases.shape[0])
    tempk = np.copy(kappa[:phases.shape[0]-1] )
    tempk = -np.flip(tempk)
    bigkappa = np.concatenate((kappa, tempk), axis=0)

    X,Y=np.meshgrid(JP, bigkappa)

    # plt.imshow(bigphase, cmap='gray', vmin=-3, vmax=3, interpolation='bilinear', extent=[-0.1, 0.1, -1, 1], aspect='auto')

    plt.contourf(X, Y, bigphase)
    plt.xlabel(r'$J_\pm/J_{zz}$')
    plt.ylabel(r'$(\kappa-1)/(\kappa+1)$')
    # plt.show()


def graphMagPhase(JP, h, phases, filename):

    X,Y = np.meshgrid(JP, h)

    plt.pcolormesh(X, Y, phases.T, vmin = 0, vmax = 16)

    plt.xlabel(r'$J_\pm/J_{y}$')
    plt.ylabel(r'$h/J_{y}$')
    plt.savefig(filename +'.png')
    plt.clf()

    # plt.contourf(X, Y, phases.T, levels=[0, 0.05,10000], colors=['#43AC63', '#B5E8C4'])
    # # plt.colorbar()
    #
    # plt.xlabel(r'$J_\pm/J_{y}$')
    # plt.ylabel(r'$h/J_{y}$')
    # plt.savefig(filename+'_split.png')
    # plt.clf()

def graphColorMesh(JP, h, phases, filename):

    X,Y = np.meshgrid(JP, h)

    plt.pcolormesh(X, Y, phases.T)
    plt.colorbar()
    plt.xlabel(r'$J_\pm/J_{y}$')
    plt.ylabel(r'$h/J_{y}$')
    plt.savefig(filename +'.png')
    plt.clf()


#region NETCDF4 file parsing

def plotLinefromnetCDF(field_direction, outfile, directory="Nia_Full_Scan/", Jpm=None, h=None, diff=False):
    if Jpm == None and h == None:
        return -1
    if (field_direction == h110).all():
        plotLine110(directory,Jpm, h, diff)
    elif (field_direction == h111).all():
        plotLine111(directory,Jpm, h, diff)
    else:
        plotLine100(directory,Jpm, h, diff)
    plt.savefig(outfile)
    plt.clf()

def readLineMFEJP(filename, dex):
    A = netCDF4.Dataset(filename)
    return A.variables['MFE'][dex,:]

def readLineMFEh(filename, dex):
    A = netCDF4.Dataset(filename)
    return A.variables['MFE'][:,dex]
def plotLine110(directory,Jpm=None, h=None,diff=False):
    if Jpm == None and h == None:
        return -1
    directString = "110_"
    filename = directory + "HanYan_"+directString+"Jpm_-0.1_0.1_h_0_0.3_"
    ftoread = [filename+"0_flux_full_info.nc", filename+"pi_flux_full_info.nc",
               filename+"pipi00_full_info.nc", filename+"00pipi_full_info.nc"]
    JPf = np.linspace(-0.1,0.1,100)
    hf = np.linspace(0,0.3,100)
    if h == None:
        dex = find_nearest(JPf, Jpm)
        if diff:
            MFEs = np.zeros((3, 100))
            zero = readLineMFEJP(ftoread[0], dex)
            plt.plot(hf, np.zeros(100))
            for i in range(3):
                MFEs[i] = readLineMFEJP(ftoread[i+1], dex)
                plt.plot(hf, MFEs[i]-zero)
            plt.legend(["0 flux", r"$\pi$ flux", r"$\pi\pi 0 0$", r"$ 00\pi\pi $"])
        else:
            MFEs = np.zeros((4, 100))
            for i in range(4):
                MFEs[i] = readLineMFEJP(ftoread[i], dex)
                plt.plot(hf, MFEs[i])
            plt.legend(["0 flux", r"$\pi$ flux", r"$\pi\pi 0 0$", r"$ 00\pi\pi $"])
    else:
        dex = find_nearest(hf, h)
        if diff:
            MFEs = np.zeros((3, 100))
            zero = readLineMFEh(ftoread[0], dex)
            plt.plot(-JPf/2, np.zeros(100))
            for i in range(3):
                MFEs[i] = readLineMFEh(ftoread[i+1], dex)
                plt.plot(-JPf/2, MFEs[i]-zero)
            plt.legend(["0 flux", r"$\pi$ flux", r"$\pi\pi 0 0$", r"$ 00\pi\pi $"])
        else:
            MFEs = np.zeros((4, 100))
            for i in range(4):
                MFEs[i] = readLineMFEh(ftoread[i], dex)
                plt.plot(-JPf/2, MFEs[i])
            plt.legend(["0 flux", r"$\pi$ flux", r"$\pi\pi 0 0$", r"$ 00\pi\pi $"])
def plotLine111(directory,Jpm=None, h=None,diff=False):
    if Jpm == None and h == None:
        return -1
    directString = "111_"
    filename = directory+"HanYan_"+directString+"Jpm_-0.1_0.1_h_0_0.3_"
    ftoread = [filename+"0_flux_full_info.nc", filename+"pi_flux_full_info.nc"]
    JPf = np.linspace(-0.1,0.1,100)
    hf = np.linspace(0,0.3,100)
    if h == None:
        dex = find_nearest(JPf, Jpm)
        if diff:
            zero = readLineMFEJP(ftoread[0], dex)
            plt.plot(hf, np.zeros(100))
            pi = readLineMFEJP(ftoread[1], dex)
            plt.plot(hf, pi-zero)
            plt.legend(["0 flux", r"$\pi$ flux"])
        else:
            MFEs = np.zeros((2, 100))
            for i in range(2):
                MFEs[i] = readLineMFEJP(ftoread[i], dex)
                plt.plot(hf, MFEs[i])
            plt.legend(["0 flux", r"$\pi$ flux"])
    else:
        dex = find_nearest(hf, h)
        if diff:
            zero = readLineMFEh(ftoread[0], dex)
            plt.plot(-JPf/2, np.zeros(100))
            pi = readLineMFEh(ftoread[1], dex)
            plt.plot(-JPf/2, pi-zero)
            plt.legend(["0 flux", r"$\pi$ flux"])
        else:
            MFEs = np.zeros((2, 100))
            for i in range(2):
                MFEs[i] = readLineMFEh(ftoread[i], dex)
                plt.plot(-JPf/2, MFEs[i])
            plt.legend(["0 flux", r"$\pi$ flux"])
def plotLine100(directory,Jpm=None, h=None,diff=False):
    if Jpm == None and h == None:
        return -1
    directString = "100_"
    filename = directory+"HanYan_"+directString+"Jpm_-0.1_0.1_h_0_0.3_"
    ftoread = [filename+"0_flux_full_info.nc", filename+"pi_flux_full_info.nc",
               filename+"0pipi0_full_info.nc", filename+"pi00pi_full_info.nc"]
    JPf = np.linspace(-0.1,0.1,100)
    hf = np.linspace(0,0.3,100)
    if h == None:
        dex = find_nearest(JPf, Jpm)
        if diff:
            MFEs = np.zeros((3, 100))
            zero = readLineMFEJP(ftoread[0], dex)
            plt.plot(hf, np.zeros(100))
            for i in range(3):
                MFEs[i] = readLineMFEJP(ftoread[i+1], dex)
                plt.plot(hf, MFEs[i]-zero)
            plt.legend(["0 flux", r"$\pi$ flux", r"$ 0\pi\pi 0$", r"$ \pi 00\pi $"])
        else:
            MFEs = np.zeros((4, 100))
            for i in range(4):
                MFEs[i] = readLineMFEJP(ftoread[i], dex)
                plt.plot(hf, MFEs[i])
            plt.legend(["0 flux", r"$\pi$ flux", r"$ 0\pi\pi 0$", r"$ \pi 00\pi $"])
    else:
        dex = find_nearest(hf, h)
        if diff:
            MFEs = np.zeros((3, 100))
            zero = readLineMFEh(ftoread[0], dex)
            plt.plot(-JPf/2, np.zeros(100))
            for i in range(3):
                MFEs[i] = readLineMFEh(ftoread[i+1], dex)
                plt.plot(-JPf/2, MFEs[i]-zero)
            plt.legend(["0 flux", r"$\pi$ flux", r"$ 0\pi\pi 0$", r"$ \pi 00\pi $"])
        else:
            MFEs = np.zeros((4, 100))
            for i in range(4):
                MFEs[i] = readLineMFEh(ftoread[i], dex)
                plt.plot(-JPf/2, MFEs[i])
            plt.legend(["0 flux", r"$\pi$ flux", r"$ 0\pi\pi 0$", r"$ \pi 00\pi $"])

#endregion


#region Phase for Magnetic Field

def findPhaseMag110(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename, Jxx=False, Jpmpm=0):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = np.mgrid[JPm:JPmax:1j*nK, hm:hmax:1j*nH].reshape(2,-1).T
    le = nK*nH
    nb = le/size

    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK


    currJH = JH[leftK:rightK]


    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp4 = np.zeros(currsizeK, dtype=np.float64)
    # sendtemp5 = np.zeros(currsizeK, dtype='U48')


    rectemp = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None
    # rectemp5 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.float64)
        rectemp4 = np.zeros(le, dtype=np.float64)
        # rectemp5 = np.zeros(le, dtype='U48')

    for i in range(currsizeK):
        if Jxx == True:
            py0s = pycon.piFluxSolver(1, -2 * currJH[i][0] + 2*Jpmpm, -2 * currJH[i][0] - 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa,
                                      BZres=BZres, flux=np.zeros(4))
            pyps = pycon.piFluxSolver(1, -2 * currJH[i][0]+ 2*Jpmpm, -2 * currJH[i][0] - 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa,
                                      BZres=BZres, flux=np.ones(4) * np.pi)
            pyp0 = pycon.piFluxSolver(1, -2 * currJH[i][0]+ 2*Jpmpm, -2 * currJH[i][0] - 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa,
                                      BZres=BZres, flux=pzzp)
            pyzp = pycon.piFluxSolver(1, -2 * currJH[i][0]+ 2*Jpmpm, -2 * currJH[i][0] - 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa,
                                      BZres=BZres, flux=zppz)
        else:
            py0s = pycon.piFluxSolver(-2*currJH[i][0] - 2*Jpmpm, 1, -2*currJH[i][0] + 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
            pyps = pycon.piFluxSolver(-2*currJH[i][0] - 2*Jpmpm, 1, -2*currJH[i][0] + 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
            pyp0 = pycon.piFluxSolver(-2*currJH[i][0] - 2*Jpmpm, 1, -2*currJH[i][0] + 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=pzzp)
            pyzp = pycon.piFluxSolver(-2*currJH[i][0] - 2*Jpmpm, 1, -2*currJH[i][0] + 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=zppz)

        py0s.solvemeanfield()
        pyps.solvemeanfield()
        pyp0.solvemeanfield()
        pyzp.solvemeanfield()

        GS = np.array([py0s.MFE(), pyps.MFE(), pyp0.MFE(),pyzp.MFE()])
        a = np.argmin(GS)
        # print(GS, a)
        if a == 0:
            sendtemp[i] = py0s.condensed
            sendtemp2[i] = GS[a]
            sendtemp3[i] = py0s.lams[0]
            sendtemp4[i] = py0s.magnetization()
            # sendtemp5[i] = np.array2string(py0s.qmin)
        elif a == 1:
            sendtemp[i] = pyps.condensed + 5
            sendtemp2[i] = GS[a]
            sendtemp3[i] = pyps.lams[0]
            sendtemp4[i] = pyps.magnetization()
            # sendtemp5[i] = np.array2string(pyps.qmin)
        elif a == 2:
            sendtemp[i] = pyp0.condensed + 10
            sendtemp2[i] = GS[a]
            sendtemp3[i] = pyp0.lams[0]
            sendtemp4[i] = pyp0.magnetization()
        else:
            sendtemp[i] = pyzp.condensed + 15
            sendtemp2[i] = GS[a]
            sendtemp3[i] = pyzp.lams[0]
            sendtemp4[i] = pyzp.magnetization()


    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0], 0))
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0], 0))
    # sendcounts5 = np.array(comm.gather(sendtemp5.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    # comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)


    if rank == 0:
        rectemp = rectemp.reshape((nK, nH))
        rectemp2 = rectemp2.reshape((nK, nH))
        rectemp3 = rectemp3.reshape((nK, nH))
        rectemp4 = rectemp4.reshape((nK, nH))
        # rectemp5 = rectemp5.reshape((nK, nH))
        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_MFE.txt', rectemp2)
        np.savetxt('Files/' + filename + '_lam.txt', rectemp3)
        np.savetxt('Files/' + filename + '_mag.txt', rectemp4)

        JP = np.linspace(JPm, JPmax, nK)
        h = np.linspace(hm, hmax, nH)
        graphMagPhase(JP, h, rectemp, 'Files/' + filename)
        graphColorMesh(JP, h, rectemp2,'Files/' + filename + '_MFE')
        graphColorMesh(JP, h, rectemp3,'Files/' + filename + '_lam')
        graphColorMesh(JP, h, rectemp4,'Files/' + filename + '_mag')
        # np.savetxt('Files/' + filename + '_q_condensed.txt', rectemp5,fmt="%s")
def findPhaseMag111(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename, Jxx=False, Jpmpm=0):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = np.mgrid[JPm:JPmax:1j*nK, hm:hmax:1j*nH].reshape(2,-1).T
    le = nK*nH
    nb = le/size

    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK


    currJH = JH[leftK:rightK]


    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp4 = np.zeros(currsizeK, dtype=np.float64)


    rectemp = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.float64)
        rectemp4 = np.zeros(le, dtype=np.float64)
        # rectemp5 = np.zeros(le, dtype='<U12')

    for i in range(currsizeK):
        if Jxx==True:
            py0s = pycon.piFluxSolver(1, -2 * currJH[i][0] + 2*Jpmpm, -2 * currJH[i][0] - 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
            pyps = pycon.piFluxSolver(1, -2 * currJH[i][0] + 2*Jpmpm, -2 * currJH[i][0] - 2*Jpmpm, n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
        else:
            py0s = pycon.piFluxSolver(-2*currJH[i][0]- 2*Jpmpm, 1, -2*currJH[i][0] + 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
            pyps = pycon.piFluxSolver(-2*currJH[i][0]- 2*Jpmpm, 1, -2*currJH[i][0] + 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)

        py0s.solvemeanfield()
        pyps.solvemeanfield()

        GS = np.array([py0s.GS(), pyps.GS()])
        a = np.argmin(GS)
        # print(GS, a)
        if a == 0:
            sendtemp[i] = py0s.condensed
            sendtemp2[i] = GS[a]
            sendtemp3[i] = py0s.lams[0]
            sendtemp4[i] = py0s.magnetization()
            # sendtemp5[i] = np.array2string(py0s.qmin)
        else:
            sendtemp[i] = pyps.condensed + 5
            sendtemp2[i] = GS[a]
            sendtemp3[i] = pyps.lams[0]
            sendtemp4[i] = pyps.magnetization()
            # sendtemp5[i] = np.array2string(pyps.qmin)
        # end = time.time()
        # print("This iteration costs " + str(end - start))

    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0], 0))
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0], 0))
    # sendcounts5 = np.array(comm.gather(sendtemp5.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    # comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)

    # comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    if rank == 0:
        rectemp = rectemp.reshape((nK, nH))
        rectemp2 = rectemp2.reshape((nK, nH))
        rectemp3 = rectemp3.reshape((nK, nH))
        rectemp4 = rectemp4.reshape((nK, nH))
        # rectemp5 = rectemp5.reshape((nK, nH))
        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_MFE.txt', rectemp2)
        np.savetxt('Files/' + filename + '_lam.txt', rectemp3)
        np.savetxt('Files/' + filename + '_mag.txt', rectemp4)

        JP = np.linspace(JPm, JPmax, nK)
        h = np.linspace(hm, hmax, nH)
        graphMagPhase(JP, h, rectemp, 'Files/' + filename)
        graphColorMesh(JP, h, rectemp2,'Files/' + filename + '_MFE')
        graphColorMesh(JP, h, rectemp3,'Files/' + filename + '_lam')
        graphColorMesh(JP, h, rectemp4,'Files/' + filename + '_mag')
        # np.savetxt('Files/' + filename + '_q_condensed.txt', rectemp5, fmt="%s")
def findPhaseMag_separate(JPm, JPmax, nK, hm, hmax, nH, n, flux, BZres, kappa, filename, Jxx=False, Jpmpm=0, FF=False, theta=0):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = np.mgrid[JPm:JPmax:1j*nK, hm:hmax:1j*nH].reshape(2,-1).T
    le = nK*nH
    nb = le/size

    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK


    currJH = JH[leftK:rightK]


    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp4 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp5 = np.zeros(currsizeK, dtype=np.float64)

    rectemp = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None
    rectemp5 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.float64)
        rectemp4 = np.zeros(le, dtype=np.float64)
        rectemp5 = np.zeros(le, dtype=np.float64)

    for i in range(currsizeK):
        if Jxx == True:
            py0s = pycon.piFluxSolver(1, -2 * currJH[i][0] + 2*Jpmpm, -2 * currJH[i][0] - 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa,
                                      BZres=BZres, flux=flux, FF=FF, theta=theta)
        else:
            py0s = pycon.piFluxSolver(-2*currJH[i][0] - 2*Jpmpm, 1, -2*currJH[i][0] + 2*Jpmpm, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=flux, FF=FF, theta=theta)
        warnings.filterwarnings('error')
        try:
            py0s.solvemeanfield()
            sendtemp[i] = py0s.condensed
            sendtemp2[i] = py0s.MFE()
            sendtemp3[i] = (py0s.xi<=1e-8).all()
            sendtemp4[i] = (py0s.chi<=1e-8).all()
            sendtemp5[i] = py0s.magnetization()
        except:
            sendtemp[i] = True
            sendtemp2[i] = np.nan
            sendtemp3[i] = np.nan
            sendtemp4[i] = np.nan
            sendtemp5[i] = np.nan
        warnings.resetwarnings()

    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0], 0))
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0], 0))
    sendcounts5 = np.array(comm.gather(sendtemp5.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)


    if rank == 0:   
        rectemp = rectemp.reshape((nK, nH))
        rectemp2 = rectemp2.reshape((nK, nH))
        rectemp3 = rectemp3.reshape((nK, nH))
        rectemp4 = rectemp4.reshape((nK, nH))
        rectemp5 = rectemp5.reshape((nK, nH))
        np.savetxt(filename+'.txt', rectemp)
        np.savetxt(filename + '_MFE.txt', rectemp2)
        np.savetxt(filename + '_xi.txt', rectemp3)
        np.savetxt(filename + '_chi.txt', rectemp4)
        np.savetxt(filename + '_mag.txt', rectemp5)

        # JP = np.linspace(JPm, JPmax, nK)
        # h = np.linspace(hm, hmax, nH)
        # graphMagPhase(JP, h, rectemp, 'Files/' + filename)
        # graphColorMesh(JP, h, rectemp2,'Files/' + filename + '_MFE')
        # graphColorMesh(JP, h, rectemp3,'Files/' + filename + '_lam')
        # graphColorMesh(JP, h, rectemp4,'Files/' + filename + '_mag')
        # np.savetxt('Files/' + filename + '_q_condensed.txt', rectemp5,fmt="%s")

def PhaseMag110_linescan(JPm, hm, hmax, nH, n, BZres, kappa, filename, Jxx=False, Jpmpm=0):

    h = np.linspace(hm, hmax,nH)

    MFE = np.zeros(nH)
    condensed = np.zeros(nH)
    XI = np.zeros(nH)
    CHI = np.zeros(nH)
    for i in range(nH):
        if Jxx == True:
            py0s = pycon.piFluxSolver(1, -2 * JPm + 2*Jpmpm, -2 * JPm - 2*Jpmpm, h=h[i], n=n, kappa=kappa,
                                      BZres=BZres, flux=np.zeros(4))
            pyps = pycon.piFluxSolver(1, -2 * JPm+ 2*Jpmpm, -2 * JPm - 2*Jpmpm, h=h[i], n=n, kappa=kappa,
                                      BZres=BZres, flux=np.ones(4) * np.pi)
            pyp0 = pycon.piFluxSolver(1, -2 * JPm+ 2*Jpmpm, -2 * JPm - 2*Jpmpm, h=h[i], n=n, kappa=kappa,
                                      BZres=BZres, flux=pzzp)
            pyzp = pycon.piFluxSolver(1, -2 * JPm+ 2*Jpmpm, -2 * JPm - 2*Jpmpm, h=h[i], n=n, kappa=kappa,
                                      BZres=BZres, flux=zppz)
        else:
            py0s = pycon.piFluxSolver(-2*JPm - 2*Jpmpm, 1, -2*JPm + 2*Jpmpm, h=h[i], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
            pyps = pycon.piFluxSolver(-2*JPm - 2*Jpmpm, 1, -2*JPm + 2*Jpmpm, h=h[i], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
            pyp0 = pycon.piFluxSolver(-2*JPm - 2*Jpmpm, 1, -2*JPm + 2*Jpmpm, h=h[i], n=n, kappa=kappa, BZres=BZres, flux=pzzp)
            pyzp = pycon.piFluxSolver(-2*JPm - 2*Jpmpm, 1, -2*JPm + 2*Jpmpm, h=h[i], n=n, kappa=kappa, BZres=BZres, flux=zppz)

        py0s.solvemeanfield()
        pyps.solvemeanfield()
        pyp0.solvemeanfield()
        pyzp.solvemeanfield()

        GS = np.array([py0s.MFE(), pyps.MFE(), pyp0.MFE(),pyzp.MFE()])
        temppy = np.array([py0s, pyps, pyp0, pyzp])

        a = np.argmin(GS)
        MFE[i] = GS[a]
        XI[i] = (temppy[a].xi <= 1e-8).all()
        CHI[i] = (temppy[a].chi<= 1e-8).all()
        condensed[i] = temppy[a].condensed + a*6
    np.savetxt(filename+".txt", condensed)
    np.savetxt(filename+"_MFE.txt", MFE)
    np.savetxt(filename+"_XI.txt", XI)
    np.savetxt(filename+"_CHI.txt", CHI)
    return condensed, MFE, XI, CHI

def PhaseMag111_linescan(JPm, hm, hmax, nH, n, BZres, kappa, filename, Jxx=False, Jpmpm=0):

    h = np.linspace(hm, hmax,nH)

    MFE = np.zeros(nH)
    condensed = np.zeros(nH)
    XI = np.zeros(nH)
    CHI = np.zeros(nH)
    for i in range(nH):
        if Jxx == True:
            py0s = pycon.piFluxSolver(1, -2 * JPm + 2*Jpmpm, -2 * JPm - 2*Jpmpm, h=h[i], n=n, kappa=kappa,
                                      BZres=BZres, flux=np.zeros(4))
            pyps = pycon.piFluxSolver(1, -2 * JPm+ 2*Jpmpm, -2 * JPm - 2*Jpmpm, h=h[i], n=n, kappa=kappa,
                                      BZres=BZres, flux=np.ones(4) * np.pi)
        else:
            py0s = pycon.piFluxSolver(-2*JPm - 2*Jpmpm, 1, -2*JPm + 2*Jpmpm, h=h[i], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
            pyps = pycon.piFluxSolver(-2*JPm - 2*Jpmpm, 1, -2*JPm + 2*Jpmpm, h=h[i], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
        py0s.solvemeanfield()
        pyps.solvemeanfield()


        GS = np.array([py0s.MFE(), pyps.MFE()])
        temppy = np.array([py0s, pyps])

        a = np.argmin(GS)
        MFE[i] = GS[a]
        XI[i] = (temppy[a].xi <= 1e-8).all()
        CHI[i] = (temppy[a].chi <= 1e-8).all()
        condensed[i] = temppy[a].condensed + a*6

    np.savetxt(filename+".txt", condensed)
    np.savetxt(filename+"_MFE.txt", MFE)
    np.savetxt(filename+"_XI.txt", XI)
    np.savetxt(filename+"_CHI.txt", CHI)
    return condensed, MFE, XI, CHI

def completeSpan(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, flux, filename, observables=False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = np.mgrid[JPm:JPmax:1j*nK, hm:hmax:1j*nH].reshape(2,-1).T
    le = nK*nH
    nb = le/size

    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK


    currJH = JH[leftK:rightK]


    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp5 = np.zeros((currsizeK,minLamK, 3), dtype=np.float64)

    rectemp = None
    rectemp2 = None
    rectemp3 = None
    rectemp5 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.float64)
        rectemp5 = np.zeros((le, minLamK, 3), dtype=np.float64)

    for i in range(currsizeK):
        py0s = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=flux)
        py0s.solvemeanfield()
        sendtemp[i] = py0s.condensed
        sendtemp2[i] = py0s.MFE()
        sendtemp3[i] = py0s.lams[0]
        temp = py0s.qmin
        leng = len(temp)
        sendtemp5[i,0:leng] = py0s.qmin


    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0], 0))
    sendcounts5 = np.array(comm.gather(sendtemp5.shape[0]*sendtemp5.shape[1]*sendtemp5.shape[2], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)


    if rank == 0:
        rectemp = rectemp.reshape((nK, nH))
        rectemp2 = rectemp2.reshape((nK, nH))
        rectemp3 = rectemp3.reshape((nK, nH))
        rectemp5 = rectemp5.reshape((nK, nH, minLamK, 3))

        ncfilename = 'Files/' + filename + '_full_info.nc'
        with nc.Dataset(ncfilename, "w") as dataset:
            dataset.createDimension("Jpm", nK)
            dataset.createDimension("h", nH)
            dataset.createDimension("index", minLamK)
            dataset.createDimension("xyz", 3)
            dataset.createDimension("dummy", 2)

            temp_var = dataset.createVariable("Jpm_H", "f4", ("Jpm", "h", "dummy"))
            temp_var[:, :, :] = JH.reshape((nK,nH,2))
            temp_var.long_name = "Jpm and H"
            temp_var1 = dataset.createVariable("q_condensed", "f4", ("Jpm", "h", "index", "xyz"))
            temp_var1[:, :, :] = rectemp5
            temp_var1.long_name = "Condensed Wave Vectors"
            temp_var2 = dataset.createVariable("lams", "f4", ("Jpm", "h"))
            temp_var2[:, :] = rectemp3
            temp_var2.long_name = "lambda"
            temp_var3 = dataset.createVariable("MFE", "f4", ("Jpm", "h"))
            temp_var3[:, :] = rectemp2
            temp_var3.long_name = "Variational Energy"
            temp_var4 = dataset.createVariable("condensed", "f4", ("Jpm", "h"))
            temp_var4[:, :] = rectemp
            temp_var4.long_name = "isCondensed"

def findPhaseMag_simple(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = np.mgrid[JPm:JPmax:1j*nK, hm:hmax:1j*nH].reshape(2,-1).T
    le = nK*nH
    nb = le/size

    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK


    currJH = JH[leftK:rightK]


    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp4 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp5 = np.zeros((currsizeK,3), dtype=np.float64)


    rectemp = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None
    rectemp5 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.float64)
        rectemp4 = np.zeros(le, dtype=np.float64)
        rectemp5 = np.zeros((le, 3), dtype=np.float64)

    for i in range(currsizeK):
        # start = time.time()
        # print(currJH[i])
        py0s = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
        pyps = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
        py0s.solvemeanfield()
        pyps.solvemeanfield()
        GS = np.array([py0s.MFE(), pyps.MFE()])
        a = np.argmin(GS)
        # print(GS, a)
        if a == 0:
            sendtemp[i] = py0s.condensed + 5*a
            sendtemp2[i] = GS[a]
            sendtemp3[i] = py0s.lams[0]
            sendtemp4[i] = py0s.magnetization()
            sendtemp5[i] = py0s.qmin
        elif a == 1:
            sendtemp[i] = pyps.condensed + 5*a
            sendtemp2[i] = GS[a]
            sendtemp3[i] = pyps.lams[0]
            sendtemp4[i] = pyps.magnetization()
            sendtemp5[i] = pyps.qmin
        # end = time.time()
        # print("This iteration costs " + str(end - start))
#


    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0], 0))
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0], 0))
    sendcounts5 = np.array(comm.gather(sendtemp5.shape[0]*sendtemp5.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)

    # comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    if rank == 0:
        rectemp = rectemp.reshape((nK, nH))
        rectemp2 = rectemp2.reshape((nK, nH))
        rectemp3 = rectemp3.reshape((nK, nH))
        rectemp4 = rectemp4.reshape((nK, nH))
        rectemp5 = rectemp5.reshape((nK, nH, 3))
        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_MFE.txt', rectemp2)
        np.savetxt('Files/' + filename + '_lam.txt', rectemp3)
        np.savetxt('Files/' + filename + '_mag.txt', rectemp4)

        ncfilename = 'Files/' + filename + '_q_condensed.nc'
        with nc.Dataset(ncfilename, "w") as dataset:
            # Create dimensions
            dataset.createDimension("Jpm", nK)
            dataset.createDimension("h", nH)
            dataset.createDimension("xyz", 3)
            temp_var = dataset.createVariable("q_condensed", "f4", ("Jpm", "h", "xyz"))
            # Assign data to variables
            temp_var[:, :, :] = rectemp5
            # Add attributes
            temp_var.long_name = "Condensed Wave Vectors"


        JP = np.linspace(JPm, JPmax, nK)
        h = np.linspace(hm, hmax, nH)
        graphMagPhase(JP, h, rectemp, 'Files/' + filename)
        graphColorMesh(JP, h, rectemp2,'Files/' + filename + '_MFE')
        graphColorMesh(JP, h, rectemp3,'Files/' + filename + '_lam')
        graphColorMesh(JP, h, rectemp4,'Files/' + filename + '_mag')

def findXYZPhase(JPm, JPmax, JP1m, JP1max, nK, BZres, kappa, filename):
    # totaltask = nK*nH
    # increment = totaltask/50
    # count = 0
    #
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = XYZparambuilder(JPm, JPmax, JP1m, JP1max, nK)
    le = len(JH)
    nb = le/size

    leftK = int(rank * nb)
    rightK = int((rank + 1) * nb)
    currsizeK = rightK - leftK


    currJH = JH[leftK:rightK]

    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.complex128)
    sendtemp4 = np.zeros(currsizeK, dtype=np.complex128)

    rectemp = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.complex128)
        rectemp4 = np.zeros(le, dtype=np.complex128)

    for i in range (currsizeK):
        py0s = pycon.piFluxSolver(currJH[i][0], 1, currJH[i][1], kappa=kappa, BZres=BZres, flux=np.zeros(4))
        py0s.solvemeanfield()
        pyps = pycon.piFluxSolver(currJH[i][0], 1, currJH[i][1], kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
        pyps.solvemeanfield()
        GS = py0s.MFE()
        GSp = pyps.MFE()
        if GS < GSp:
            sendtemp[i] = py0s.condensed or (py0s.chi>1e-12).all()
            sendtemp2[i] = GS
            sendtemp3[i] = (py0s.xi<=1e-12).all()
            sendtemp4[i] = (py0s.chi<=1e-12).all()
        else:
            sendtemp[i] = (pyps.condensed or (py0s.chi>1e-12).all()) + 5
            sendtemp2[i] = GSp
            sendtemp3[i] = (pyps.xi<=1e-12).all()
            sendtemp4[i] = (pyps.chi<=1e-12).all()


    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 =  np.array(comm.gather(sendtemp3.shape[0], 0))
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)

    if rank == 0:
        rectemp = inverseXYZparambuilder(rectemp.reshape((int(nK/2),nK+1)))
        rectemp2 = inverseXYZparambuilder(rectemp2.reshape((int(nK/2),nK+1)))
        rectemp3 = inverseXYZparambuilder(rectemp3.reshape((int(nK/2),nK+1)))
        rectemp4 = inverseXYZparambuilder(rectemp4.reshape((int(nK/2),nK+1)))


        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_MFE.txt', rectemp2)
        np.savetxt('Files/' + filename + '_xi.txt', rectemp3)
        np.savetxt('Files/' + filename + '_chi.txt', rectemp4)

        JP = np.linspace(JPm, JPmax, nK)
        JP1 = np.linspace(JP1m, JP1max, nK)

        graphMagPhase(JP, JP1, rectemp, 'Files/' + filename)
        graphColorMesh(JP, JP1, rectemp2,'Files/' + filename + '_MFE')
        graphColorMesh(JP, JP1, rectemp3,'Files/' + filename + '_xi')
        graphColorMesh(JP, JP1, rectemp4,'Files/' + filename + '_chi')


def findXYZPhase_separate(JPm, JPmax, JP1m, JP1max, nK, BZres, kappa, flux, filename, *args, symmetrized=True):
    # totaltask = nK*nH
    # increment = totaltask/50
    # count = 0
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if symmetrized:
        JH = XYZparambuilder(JPm, JPmax, JP1m, JP1max, nK)
    else:
        JH = np.mgrid[JPm:JPmax:1j * nK, JP1m:JP1max:1j * nK].reshape(2,-1).T

    le = len(JH)
    nb = le/size

    leftK = int(rank * nb)
    rightK = int((rank + 1) * nb)
    currsizeK = rightK - leftK


    currJH = JH[leftK:rightK]

    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.complex128)
    sendtemp4 = np.zeros(currsizeK, dtype=np.complex128)

    rectemp = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.complex128)
        rectemp4 = np.zeros(le, dtype=np.complex128)

    for i in range (currsizeK):
        py0s = pycon.piFluxSolver(currJH[i][0], 1, currJH[i][1], *args, kappa=kappa, BZres=BZres, flux=flux)
        warnings.filterwarnings('error')
        try:
            py0s.solvemeanfield()
            sendtemp[i] = py0s.condensed
            sendtemp2[i] = py0s.MFE()
            sendtemp3[i] = (py0s.xi<=1e-8).all()
            sendtemp4[i] = (py0s.chi<=1e-8).all()
        except:
            sendtemp[i] = np.nan
            sendtemp2[i] = np.nan
            sendtemp3[i] = np.nan
            sendtemp4[i] = np.nan
        warnings.resetwarnings()
    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 =  np.array(comm.gather(sendtemp3.shape[0], 0))
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)

    if rank == 0:
        if symmetrized:
            rectemp = inverseXYZparambuilder(rectemp.reshape((int(nK/2),nK+1)))
            rectemp2 = inverseXYZparambuilder(rectemp2.reshape((int(nK/2),nK+1)))
            rectemp3 = inverseXYZparambuilder(rectemp3.reshape((int(nK/2),nK+1)))
            rectemp4 = inverseXYZparambuilder(rectemp4.reshape((int(nK/2),nK+1)))
        else:
            rectemp = rectemp.reshape((nK, nK))
            rectemp2 = rectemp2.reshape((nK, nK))
            rectemp3 = np.real(rectemp3.reshape((nK, nK)))
            rectemp4 = np.real(rectemp4.reshape((nK, nK)))

        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_MFE.txt', rectemp2)
        np.savetxt('Files/' + filename + '_xi.txt', rectemp3)
        np.savetxt('Files/' + filename + '_chi.txt', rectemp4)

        JP = np.linspace(JPm, JPmax, nK)
        JP1 = np.linspace(JP1m, JP1max, nK)

        graphMagPhase(JP, JP1, rectemp, 'Files/' + filename)
        graphColorMesh(JP, JP1, rectemp2,'Files/' + filename + '_MFE')
        graphColorMesh(JP, JP1, rectemp3,'Files/' + filename + '_xi')
        graphColorMesh(JP, JP1, rectemp4,'Files/' + filename + '_chi')
        
def findXYZPhase_separate_unconstrained(JPm, JPmax, JP1m, JP1max, nK, BZres, kappa, flux, filename):
    # totaltask = nK*nH
    # increment = totaltask/50
    # count = 0
    #
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = XYZparambuilder(JPm, JPmax, JP1m, JP1max, nK)
    le = len(JH)
    nb = le/size

    leftK = int(rank * nb)
    rightK = int((rank + 1) * nb)
    currsizeK = rightK - leftK


    currJH = JH[leftK:rightK]

    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.complex128)
    # sendtemp4 = np.zeros(currsizeK, dtype=np.complex128)

    rectemp = None
    rectemp2 = None
    rectemp3 = None
    # rectemp4 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.complex128)
        # rectemp4 = np.zeros(le, dtype=np.complex128)

    for i in range (currsizeK):
        py0s = pycon.piFluxSolver(currJH[i][0], 1, currJH[i][1], kappa=kappa, BZres=BZres, flux=flux, unconstrained=True)
        warnings.filterwarnings('error')
        try:
            py0s.solvemeanfield()
            sendtemp[i] = py0s.condensed
            sendtemp2[i] = py0s.MFE()
            sendtemp3[i] = (py0s.xi<=1e-6).all()
        except:
            sendtemp[i] = np.nan
            sendtemp2[i] = np.nan
            sendtemp3[i] = np.nan
        warnings.resetwarnings()

    
    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 =  np.array(comm.gather(sendtemp3.shape[0], 0))
    # sendcounts4 = np.array(comm.gather(sendtemp4.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    # comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)

    if rank == 0:
        rectemp = inverseXYZparambuilder(rectemp.reshape((int(nK/2),nK+1)))
        rectemp2 = inverseXYZparambuilder(rectemp2.reshape((int(nK/2),nK+1)))
        rectemp3 = inverseXYZparambuilder(rectemp3.reshape((int(nK/2),nK+1)))


        # rectemp4 = rectemp4.reshape((nK,nK))

        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_MFE.txt', rectemp2)
        np.savetxt('Files/' + filename + '_xi.txt', rectemp3)
        # np.savetxt('Files/' + filename + '_chi.txt', rectemp4)

        JP = np.linspace(JPm, JPmax, nK)
        JP1 = np.linspace(JP1m, JP1max, nK)

        graphMagPhase(JP, JP1, rectemp, 'Files/' + filename)
        graphColorMesh(JP, JP1, rectemp2,'Files/' + filename + '_MFE')
        graphColorMesh(JP, JP1, rectemp3,'Files/' + filename + '_xi')
        # graphColorMesh(JP, JP1, rectemp4,'Files/' + filename + '_chi')



def conclude_XYZ_0_field(filename, Jpmin, Jpmax):

    A1 = filename+"_0_flux"
    A2 = filename+"_0_flux_nS=1"
    A3 = filename+"_pi_flux"
    A4 = filename+"_pi_flux_nS=1"


    D1 = np.loadtxt(A1+"_MFE.txt")
    D2 = np.loadtxt(A2+"_MFE.txt")
    D3 = np.loadtxt(A3+"_MFE.txt")
    D4 = np.loadtxt(A4+"_MFE.txt")

    D = np.array([D1,D2,D3,D4])

    X1 = np.loadtxt(A1+"_xi.txt", dtype=np.complex128)
    X2 = np.loadtxt(A2+"_xi.txt", dtype=np.complex128)
    X3 = np.loadtxt(A3+"_xi.txt", dtype=np.complex128)
    X4 = np.loadtxt(A4+"_xi.txt", dtype=np.complex128)

    X = np.array([X1,X2,X3,X4])

    C1 = np.loadtxt(A1+".txt")
    C2 = np.loadtxt(A2+".txt")
    C3 = np.loadtxt(A3+".txt")
    C4 = np.loadtxt(A4+".txt")

    C = np.array([C1,C2,C3,C4])

    Ch1 = np.loadtxt(A1+"_chi.txt", dtype=np.complex128)
    Ch2 = np.loadtxt(A2+"_chi.txt", dtype=np.complex128)
    Ch3 = np.loadtxt(A3+"_chi.txt", dtype=np.complex128)
    Ch4 = np.loadtxt(A4+"_chi.txt", dtype=np.complex128)

    Ch = np.array([Ch1,Ch2,Ch3,Ch4])

    phase = np.zeros((int(len(D1)/3*4), int(len(D1)/3*4)))
    phase[:] = np.nan

    xi = np.zeros((len(D1), len(D1)))
    chi = np.zeros((len(D1), len(D1)))
    MFE = np.zeros((len(D1), len(D1)))

    Jpm = np.zeros((len(D1), len(D1)))
    Jpmpm = np.zeros((len(D1), len(D1)))

    offset = len(phase)-len(D1)

    for i in range(len(D1)):
        for j in range(D1.shape[1]):
            Jxx = Jpmin+((Jpmax-Jpmin)/D1.shape[0]*(i+1))
            Jyy = Jpmin+((Jpmax-Jpmin)/D1.shape[1]*(j+1))

            nSdexi = int((Jxx+1)/2*40-1)
            nSdexj = int((Jyy+1)/2*40-1)

            Jpm[i,j] = -(Jxx+Jyy)/4
            Jpmpm[i,j] = (Jxx-Jyy)/4

            tempD = D[:,i,j]
            # tempD = np.array([D[0,i,j],D[1,nSdexi,nSdexj],D[2,i,j],D[3,i,j]])
            a = np.argmin(tempD)

            if not C[a,i,j]:
                phase[i+offset,j+offset] = a//2
                xi[i,j] = X[a, i, j]
                chi[i,j] = Ch[a, i, j]
                MFE[i, j] = D[a, i, j]
            else:
                phase[i+offset,j+offset] = np.nan
            # if not C[a,i,j]:
            #     phase[i,j] = phase[i,j] + 5*X[a,i,j]
    phase[13+offset,13+offset] = 0
    # plt.pcolormesh(Jpm, Jpmpm, phase)
    # plt.ylim([0,0.5])
    # # plt.colorbar()
    # plt.xlabel(r"$J_{\pm}/J_{yy}$")
    # plt.ylabel(r"$J_{\pm\pm}/J_{yy}$")
    # plt.savefig(filename+"Jpm_Jpmpm.pdf")
    # plt.clf()
    plt.imshow(phase.T, origin='lower', extent=[Jpmin, Jpmax, Jpmin, Jpmax], aspect='equal')
    plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+".pdf")
    plt.clf()

    plt.imshow(xi.T, origin='lower', interpolation='bilinear', extent=[Jpmin, Jpmax, Jpmin, Jpmax], aspect='equal')
    # plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+"_xi.pdf")
    plt.clf()

    plt.imshow(chi.T, origin='lower', interpolation='bilinear', extent=[Jpmin, Jpmax, Jpmin, Jpmax], aspect='equal')
    # plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+"_chi.pdf")
    plt.clf()

    plt.imshow(MFE.T, origin='lower', interpolation='bilinear', extent=[Jpmin, Jpmax, Jpmin, Jpmax], aspect='equal')
    # plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+"_MFE.pdf")
    plt.clf()


def conclude_XYZ_0_field_unconstrained(filename):
    A1 = filename+"_0_flux_unconstrained"
    A2 = filename+"_pi_flux_unconstrained"


    D1 = np.loadtxt(A1+"_MFE.txt")
    D2 = np.loadtxt(A2+"_MFE.txt")

    D = np.array([D1,D2])

    X1 = np.loadtxt(A1+"_xi.txt")
    X2 = np.loadtxt(A2+"_xi.txt")

    X = np.array([X1,X2])

    C1 = np.loadtxt(A1+".txt")
    C2 = np.loadtxt(A2+".txt")

    C = np.array([C1,C2])

    phase = np.zeros((len(D1), len(D1)))
    Jpm = np.zeros((len(D1), len(D1)))
    Jpmpm = np.zeros((len(D1), len(D1)))
    for i in range(len(D1)):
        for j in range(D1.shape[0]):
            tempD = D[:,i,j]
            a = np.argmin(tempD)
            phase[i,j] = a//2 + 5*C[a,i,j]
            if np.isnan(phase[i,j]):
                phase[i,j] = 5
            Jxx = -0.5+(1.5/80*(i+1))
            Jyy = -0.5+(1.5/80*(j+1))
            Jpm[i,j] = -(Jxx+Jyy)/4
            Jpmpm[i,j] = (Jxx-Jyy)/4
            # if not C[a,i,j]:
            #     phase[i,j] = phase[i,j] + 5*X[a,i,j]
    plt.pcolormesh(Jpm, Jpmpm, phase)
    plt.ylim([0,0.5])
    plt.colorbar()
    plt.savefig(filename+"Jpm_Jpmpm.pdf")
    plt.clf()
    plt.imshow(phase.T, origin='lower', interpolation='bilinear', extent=[-0.5, 1, -0.5, 1], aspect='auto')
    plt.colorbar()
    plt.savefig(filename+".pdf")
    plt.clf()

def conclude_XYZ_finite_field(filename, Jpmin, Jpmax, hmin, hmax):
    A = np.loadtxt(filename+'.txt')
    A = np.where(A==1, np.nan, A)
    A = np.where(A==6, np.nan, A)
    A = np.where(A==11, np.nan, A)
    A = np.where(A==16, np.nan, A)


    # plt.show()

    for i in range(len(A)):
        for j in range(A.shape[1]):
            JP = Jpmin+((Jpmax-Jpmin)/A.shape[0]*(i+1))
            if JP>0:
                A[i,j] = np.nan
    C = plt.imshow(A.T, origin='lower', aspect='auto', extent=[Jpmin, Jpmax, hmin, hmax])
    plt.xlabel(r'$J_\pm/J_{y}$')
    plt.ylabel(r'$h/J_{y}$')
    plt.savefig(filename+'.pdf')
    plt.clf()


def conclude_XYZ_0_field_job_array(filename):

    import os

    num_grids = 0
    grid_reso = 0

    for files in os.listdir(filename):
        if files.endswith("MFE.txt"):
            dump, dump1, flux_string, flux_string1, nS, job_id, dump2, dump3, total_job, dump4 = files.split("_")
            d0 = np.loadtxt(filename+"/"+files)
            num_grids = int(np.sqrt(int(total_job)/4))
            grid_reso = len(d0)
            break
    
    phase = np.zeros((num_grids*grid_reso, num_grids*grid_reso))
    MFEs = np.zeros((num_grids*grid_reso, num_grids*grid_reso))
    XIs = np.zeros((num_grids*grid_reso, num_grids*grid_reso))
    CHIs = np.zeros((num_grids*grid_reso, num_grids*grid_reso))
    print(num_grids, grid_reso)
    mult = int(total_job)/4
    for files in os.listdir(filename):
        if files.endswith("MFE.txt"):
            dump, dump1, flux_string, flux_string1, nS, job_id, dump2, dump3, total_job, dump4 = files.split("_")

            if int(job_id) <= mult:
                temp = filename+'/'+dump+"_"+dump1+"_0_flux_0.0_"+str(int(job_id))+"_out_of_"+total_job+".txt"
                temp1 = filename+'/'+dump+"_"+dump1+"_0_flux_1.0_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+".txt"
                temp2 = filename+'/'+dump+"_"+dump1+"_pi_flux_0.0_"+str(int(int(job_id)+2*mult))+"_out_of_"+total_job+".txt"
                temp3 = filename+'/'+dump+"_"+dump1+"_pi_flux_1.0_"+str(int(int(job_id)+3*mult))+"_out_of_"+total_job+".txt"

                d0 = np.loadtxt(temp)
                d1 = np.loadtxt(temp1)
                d2 = np.loadtxt(temp2)
                d3 = np.loadtxt(temp3)

                D = np.array([d0,d1,d2,d3])

                temp = filename+'/'+dump+"_"+dump1+"_0_flux_0.0_"+str(int(job_id))+"_out_of_"+total_job+"_MFE.txt"
                temp1 = filename+'/'+dump+"_"+dump1+"_0_flux_1.0_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+"_MFE.txt"
                temp2 = filename+'/'+dump+"_"+dump1+"_pi_flux_0.0_"+str(int(int(job_id)+2*mult))+"_out_of_"+total_job+"_MFE.txt"
                temp3 = filename+'/'+dump+"_"+dump1+"_pi_flux_1.0_"+str(int(int(job_id)+3*mult))+"_out_of_"+total_job+"_MFE.txt"

                MFE0 = np.loadtxt(temp)
                MFE1 = np.loadtxt(temp1)
                MFE2 = np.loadtxt(temp2)
                MFE3 = np.loadtxt(temp3)

                MFE = np.array([MFE0, MFE1, MFE2, MFE3])

                temp = filename+'/'+dump+"_"+dump1+"_0_flux_0.0_"+str(int(job_id))+"_out_of_"+total_job+"_xi.txt"
                temp1 = filename+'/'+dump+"_"+dump1+"_0_flux_1.0_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+"_xi.txt"
                temp2 = filename+'/'+dump+"_"+dump1+"_pi_flux_0.0_"+str(int(int(job_id)+2*mult))+"_out_of_"+total_job+"_xi.txt"
                temp3 = filename+'/'+dump+"_"+dump1+"_pi_flux_1.0_"+str(int(int(job_id)+3*mult))+"_out_of_"+total_job+"_xi.txt"

                XI0 = np.loadtxt(temp)
                XI1 = np.loadtxt(temp1)
                XI2 = np.loadtxt(temp2)
                XI3 = np.loadtxt(temp3)

                XI = np.array([XI0, XI1, XI2, XI3])

                temp = filename+'/'+dump+"_"+dump1+"_0_flux_0.0_"+str(int(job_id))+"_out_of_"+total_job+"_chi.txt"
                temp1 = filename+'/'+dump+"_"+dump1+"_0_flux_1.0_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+"_chi.txt"
                temp2 = filename+'/'+dump+"_"+dump1+"_pi_flux_0.0_"+str(int(int(job_id)+2*mult))+"_out_of_"+total_job+"_chi.txt"
                temp3 = filename+'/'+dump+"_"+dump1+"_pi_flux_1.0_"+str(int(int(job_id)+3*mult))+"_out_of_"+total_job+"_chi.txt"

                CHI0 = np.loadtxt(temp)
                CHI1 = np.loadtxt(temp1)
                CHI2 = np.loadtxt(temp2)
                CHI3 = np.loadtxt(temp3)

                CHI = np.array([CHI0, CHI1, CHI2, CHI3])

                JPM_PARAM_SIZE = int(total_job) / 4
                SlURM_ID = int(job_id) - 1
                flux_ind_ns_ind = SlURM_ID // JPM_PARAM_SIZE
                flux_ind = flux_ind_ns_ind//2
                nS = flux_ind_ns_ind % 2

                Jpm_section = int(np.sqrt(JPM_PARAM_SIZE))
                JPM_SECTION_ID = int(SlURM_ID) % JPM_PARAM_SIZE

                Jpm_length = int(JPM_SECTION_ID) // Jpm_section
                Jpm_width = int(JPM_SECTION_ID) % Jpm_section

                Jpm_unit = 2/Jpm_section

                Jpm_length_start = -1 + Jpm_length * Jpm_unit
                Jpm_length_end = -1 + (Jpm_length+1) * Jpm_unit
                Jpm_width_start = -1 + Jpm_width * Jpm_unit
                Jpm_width_end = -1 + (Jpm_width+1) * Jpm_unit
                offset_x = Jpm_length * grid_reso
                offset_y = Jpm_width * grid_reso

                print(Jpm_length_start, Jpm_length_end, Jpm_width_start, Jpm_width_end)

                for i in range(len(d0)):
                    for j in range(d0.shape[1]):
                        Jxx = Jpm_length_start+((Jpm_length_end-Jpm_length_start)/d0.shape[0]*(i+1))
                        Jyy = Jpm_width_start+((Jpm_width_end-Jpm_width_start)/d0.shape[1]*(j+1))
                        Jpm = -(Jxx+Jyy)/4
                        Jpmpm = (Jxx-Jyy)/4
                        tempD = MFE[:,i,j]
                        a = np.argmin(tempD)

                        if not D[a,i,j]:
                            phase[i+offset_x,j+offset_y] = a//2
                            MFEs[i+offset_x, j+offset_y] = MFE[a, i, j]
                            XIs[i+offset_x, j+offset_y] = XI[a, i, j]
                            CHIs[i+offset_x, j+offset_y] = CHI[a, i, j]
                        else:
                            phase[i+offset_x,j+offset_y] = np.nan
                        
                        # if  XI[a, i, j] == 0:
                            # phase[i+offset_x,j+offset_y] = np.nan

                        # if Jpm > 0 and phase[i+offset_x,j+offset_y] == 1:
                        #     phase[i+offset_x,j+offset_y] = np.nan

                        # if (Jxx < -0.35 or Jyy < -0.35) and phase[i+offset_x,j+offset_y] == 0:
                        #     phase[i+offset_x,j+offset_y] = np.nan

                        # if (Jxx < -0.35 or Jyy < -0.35) and phase[i+offset_x,j+offset_y] == 1 and np.abs(Jpmpm) > 0.15:
                        #     phase[i+offset_x,j+offset_y] = np.nan
                        
                        # if Jpm == 0:
                        #     phase[i+offset_x,j+offset_y] = 0


    plt.imshow(phase.T, origin='lower', extent=[-1, 1, -1, 1], aspect='equal')
    plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+"_1.pdf")
    plt.clf()

    plt.imshow(MFEs.T, origin='lower', interpolation='bilinear', extent=[-1, 1, -1, 1], aspect='equal')
    # plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+"_MFE.pdf")
    plt.clf()

    plt.imshow(XIs.T, origin='lower', interpolation='bilinear', extent=[-1, 1, -1, 1], aspect='equal')
    # plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+"_xi.pdf")
    plt.clf()

    plt.imshow(CHIs.T, origin='lower', interpolation='bilinear', extent=[-1, 1, -1, 1], aspect='equal')
    # plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+"_chi.pdf")
    plt.clf()
def conclude_XYZ_finite_field_job_array(filename):

    import os

    num_grids = 0
    grid_reso = 0

    stuff = filename.split("_")
    Jpmpm = 0.2

    for i in stuff:
        if i == "Jpmpm=0":
            Jpmpm = 0

    field_dir = stuff[3]
    Jstart = -0.5 + Jpmpm
    Jend = 0.1
    Hstart = 0
    Hend = 0.5

    if field_dir == "110":
        num_sec = 4
    else:
        num_sec = 2

    for files in os.listdir(filename):
        if files.endswith("MFE.txt"):
            dump, Jpm_start, Jpm_end, dump1, h_start, h_end, job_id, dump2, dump3, total_job, dump4 = files.split("_")
            d0 = np.loadtxt(filename+"/"+files)
            num_grids = int(np.sqrt(int(total_job)/num_sec))
            grid_reso = len(d0)
            break
    
    phase = np.zeros((num_grids*grid_reso, num_grids*grid_reso))
    MFEs = np.zeros((num_grids*grid_reso, num_grids*grid_reso))
    XIs = np.zeros((num_grids*grid_reso, num_grids*grid_reso))
    CHIs = np.zeros((num_grids*grid_reso, num_grids*grid_reso))
    print(num_grids, grid_reso)
    mult = int(total_job)/num_sec
    print(mult)
    for files in os.listdir(filename):
        if files.endswith("MFE.txt"):
            dump, Jpm_start, Jpm_end, dump1, h_start, h_end, job_id, dump2, dump3, total_job, dump4 = files.split("_")

            if int(job_id) <= mult:
                if num_sec == 4:
                    temp = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(job_id))+"_out_of_"+total_job+".txt"
                    temp1 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+".txt"
                    temp2 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+2*mult))+"_out_of_"+total_job+".txt"
                    temp3 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+3*mult))+"_out_of_"+total_job+".txt"

                    d0 = np.loadtxt(temp)
                    d1 = np.loadtxt(temp1)
                    d2 = np.loadtxt(temp2)
                    d3 = np.loadtxt(temp3)

                    D = np.array([d0,d1,d2,d3])

                    temp = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(job_id))+"_out_of_"+total_job+"_MFE.txt"
                    temp1 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+"_MFE.txt"
                    temp2 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+2*mult))+"_out_of_"+total_job+"_MFE.txt"
                    temp3 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+3*mult))+"_out_of_"+total_job+"_MFE.txt"

                    MFE0 = np.loadtxt(temp)
                    MFE1 = np.loadtxt(temp1)
                    MFE2 = np.loadtxt(temp2)
                    MFE3 = np.loadtxt(temp3)

                    MFE = np.array([MFE0, MFE1, MFE2, MFE3])


                    temp = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(job_id))+"_out_of_"+total_job+"_xi.txt"
                    temp1 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+"_xi.txt"
                    temp2 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+2*mult))+"_out_of_"+total_job+"_xi.txt"
                    temp3 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+3*mult))+"_out_of_"+total_job+"_xi.txt"

                    
                    XI0 = np.loadtxt(temp)
                    XI1 = np.loadtxt(temp1)
                    XI2 = np.loadtxt(temp2)
                    XI3 = np.loadtxt(temp3)

                    XI = np.array([XI0, XI1, XI2, XI3])

                    temp = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(job_id))+"_out_of_"+total_job+"_chi.txt"
                    temp1 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+"_chi.txt"
                    temp2 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+2*mult))+"_out_of_"+total_job+"_chi.txt"
                    temp3 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+3*mult))+"_out_of_"+total_job+"_chi.txt"

                    
                    CHI0 = np.loadtxt(temp)
                    CHI1 = np.loadtxt(temp1)
                    CHI2 = np.loadtxt(temp2)
                    CHI3 = np.loadtxt(temp3)

                    CHI = np.array([CHI0, CHI1, CHI2, CHI3])
                else:
                    temp = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(job_id))+"_out_of_"+total_job+".txt"
                    temp1 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+".txt"

                    d0 = np.loadtxt(temp)
                    d1 = np.loadtxt(temp1)

                    D = np.array([d0,d1])

                    temp = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(job_id))+"_out_of_"+total_job+"_MFE.txt"
                    temp1 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+"_MFE.txt"

                    MFE0 = np.loadtxt(temp)
                    MFE1 = np.loadtxt(temp1)


                    MFE = np.array([MFE0, MFE1])

                    temp = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(job_id))+"_out_of_"+total_job+"_xi.txt"
                    temp1 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+"_xi.txt"

                    XI0 = np.loadtxt(temp)
                    XI1 = np.loadtxt(temp1)


                    XI = np.array([XI0, XI1])

    
                    temp = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(job_id))+"_out_of_"+total_job+"_chi.txt"
                    temp1 = filename+'/'+dump+"_"+Jpm_start+"_"+Jpm_end+"_"+dump1+"_"+h_start+"_"+h_end+"_"+str(int(int(job_id)+mult))+"_out_of_"+total_job+"_chi.txt"
                    
                    CHI0 = np.loadtxt(temp)
                    CHI1 = np.loadtxt(temp1)

                    CHI = np.array([CHI0, CHI1])
                JPM_PARAM_SIZE = int(total_job) / num_sec
                SlURM_ID = int(job_id) - 1
                flux_ind_ns_ind = SlURM_ID // JPM_PARAM_SIZE

                if Jpmpm == 0 and field_dir == "111":
                    if flux_ind_ns_ind == 0:
                        flux = np.zeros(4)
                    elif flux_ind_ns_ind == 1:
                        flux = np.ones(4)*np.pi
                    elif flux_ind_ns_ind == 2:
                        flux = FFFluxGen(np.pi/3)
                        FF = True
                else:
                    if flux_ind_ns_ind == 0:
                        flux = np.zeros(4)
                    elif flux_ind_ns_ind == 1:
                        flux = np.ones(4)*np.pi
                    elif flux_ind_ns_ind == 2:
                        flux = zppz
                    else:
                        flux = pzzp

                Jpm_section = int(np.sqrt(JPM_PARAM_SIZE))
                JPM_SECTION_ID = int(SlURM_ID) % JPM_PARAM_SIZE

                Jpm_length = int(JPM_SECTION_ID) // Jpm_section
                Jpm_width = int(JPM_SECTION_ID) % Jpm_section

                # Jpm_unit = np.abs(Jend-Jstart)/Jpm_section
                # h_unit = np.abs(Hend-Hstart)/Jpm_section

                offset_x = Jpm_length * grid_reso
                offset_y = Jpm_width * grid_reso

                for i in range(len(d0)):
                    for j in range(d0.shape[1]):
                        # Jxx = Jpm_length_start+((Jpm_length_end-Jpm_length_start)/d0.shape[0]*(i+1))
                        # Jyy = Jpm_width_start+((Jpm_width_end-Jpm_width_start)/d0.shape[1]*(j+1))
                        # Jpm = -(Jxx+Jyy)/4
                        # Jpmpm = (Jxx-Jyy)/4
                        tempD = MFE[:,i,j]
                        a = np.argmin(tempD)

                        if not D[a,i,j] and CHI[a, i, j]:
                            phase[i+offset_x,j+offset_y] = a
                            MFEs[i+offset_x, j+offset_y] = MFE[a, i, j]
                            XIs[i+offset_x, j+offset_y] = XI[a, i, j]
                            CHIs[i+offset_x, j+offset_y] = CHI[a, i, j]
                        else:
                            phase[i+offset_x,j+offset_y] = np.nan
                        
                        # if  XI[a, i, j] == 0:
                            # phase[i+offset_x,j+offset_y] = np.nan

                        # if Jpm > 0 and phase[i+offset_x,j+offset_y] == 1:
                        #     phase[i+offset_x,j+offset_y] = np.nan

                        # if (Jxx < -0.35 or Jyy < -0.35) and phase[i+offset_x,j+offset_y] == 0:
                        #     phase[i+offset_x,j+offset_y] = np.nan

                        # if (Jxx < -0.35 or Jyy < -0.35) and phase[i+offset_x,j+offset_y] == 1 and np.abs(Jpmpm) > 0.15:
                        #     phase[i+offset_x,j+offset_y] = np.nan
                        
                        # if Jpm == 0:
                        #     phase[i+offset_x,j+offset_y] = 0


    plt.imshow(phase.T, origin='lower', extent=[Jstart, Jend, Hstart, Hend], aspect='auto', vmin=0, vmax=3)
    plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+".pdf")
    plt.clf()

    plt.imshow(MFEs.T, origin='lower', extent=[Jstart, Jend, Hstart, Hend], aspect='auto')
    # plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+"_MFE.pdf")
    plt.clf()

    plt.imshow(XIs.T, origin='lower', extent=[Jstart, Jend, Hstart, Hend], aspect='auto')
    # plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+"_xi.pdf")
    plt.clf()

    plt.imshow(CHIs.T, origin='lower', extent=[Jstart, Jend, Hstart, Hend], aspect='auto')
    # plt.colorbar()
    plt.xlabel(r"$J_{xx}/J_{yy}$")
    plt.ylabel(r"$J_{zz}/J_{yy}$")
    plt.savefig(filename+"_chi.pdf")
    plt.clf()
#endregion

#region Phase for Magnetic Field - Exclusive Boson
def findPhaseMag110_ex(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename, Jxx=False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = np.mgrid[JPm:JPmax:1j*nK, hm:hmax:1j*nH].reshape(2,-1).T
    le = nK*nH
    nb = le/size

    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK


    currJH = JH[leftK:rightK]


    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp1 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)

    rectemp = None
    rectemp1 = None
    rectemp2 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp1 = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)

    for i in range(currsizeK):
        try:
            if Jxx==True:
                py0s = pyex.piFluxSolver(1, -2 * currJH[i][0], -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.zeros(4))
                pyps = pyex.piFluxSolver(1, -2 * currJH[i][0], -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.ones(4) * np.pi)
                pyp0 = pyex.piFluxSolver(1, -2 * currJH[i][0], -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.array([np.pi, np.pi, 0, 0]))
                pyzp = pyex.piFluxSolver(1, -2 * currJH[i][0], -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.array([0, 0, np.pi, np.pi]))
            else:
                py0s = pyex.piFluxSolver(-2*currJH[i][0], 1, -2*currJH[i][0], h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
                pyps = pyex.piFluxSolver(-2*currJH[i][0], 1, -2*currJH[i][0], h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
                pyp0 = pyex.piFluxSolver(-2*currJH[i][0], 1, -2*currJH[i][0], h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.array([np.pi,np.pi,0,0]))
                pyzp = pyex.piFluxSolver(-2*currJH[i][0], 1, -2*currJH[i][0], h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.array([0,0,np.pi,np.pi]))
            py0s.solvemeanfield()
            pyps.solvemeanfield()
            pyp0.solvemeanfield()
            pyzp.solvemeanfield()
            GS = np.array([py0s.GS(), pyps.GS(), pyp0.GS(),pyzp.GS()])
            a = np.argmin(GS)
            sendtemp2[i] = a
            if a == 0:
                sendtemp[i] = GS[a]
                sendtemp1[i] = py0s.occu_num()
            elif a == 1:
                sendtemp[i] = GS[a]
                sendtemp1[i] = pyps.occu_num()

            elif a == 2:
                sendtemp[i] = GS[a]
                sendtemp1[i] = pyp0.occu_num()
            else:
                sendtemp[i] = GS[a]
                sendtemp1[i] = pyzp.occu_num()
        except:
            sendtemp[i] = np.NaN
            sendtemp1[i] = np.NaN
            sendtemp2[i] = np.NaN


    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    if rank == 0:
        rectemp = rectemp.reshape((nK, nH))
        rectemp1 = rectemp1.reshape((nK, nH))
        rectemp2 = rectemp2.reshape((nK, nH))
        np.savetxt('Files/' + filename+'_GS.txt', rectemp)
        np.savetxt('Files/' + filename + '_N.txt', rectemp1)
        np.savetxt('Files/' + filename + '.txt', rectemp2)
        JP = np.linspace(JPm, JPmax, nK)
        h = np.linspace(hm, hmax, nH)
        graphColorMesh(JP, h, rectemp,'Files/' + filename + '_GS')
        graphColorMesh(JP, h, rectemp1,'Files/' + filename + '_N')
        graphColorMesh(JP, h, rectemp2,'Files/' + filename)
def findPhaseMag001_ex(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename, Jxx=False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = np.mgrid[JPm:JPmax:1j * nK, hm:hmax:1j * nH].reshape(2, -1).T
    le = nK * nH
    nb = le / size

    leftK = int(rank * nb)
    rightK = int((rank + 1) * nb)
    currsizeK = rightK - leftK

    currJH = JH[leftK:rightK]

    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp1 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)

    rectemp = None
    rectemp1 = None
    rectemp2 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp1 = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)

    for i in range(currsizeK):
        try:
            if Jxx == True:
                py0s = pyex.piFluxSolver(1, -2 * currJH[i][0], -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.zeros(4))
                pyps = pyex.piFluxSolver(1, -2 * currJH[i][0], -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.ones(4) * np.pi)
                pyp0 = pyex.piFluxSolver(1, -2 * currJH[i][0], -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.array([np.pi, np.pi, 0, 0]))
                pyzp = pyex.piFluxSolver(1, -2 * currJH[i][0], -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.array([0, 0, np.pi, np.pi]))
            else:
                py0s = pyex.piFluxSolver(-2 * currJH[i][0], 1, -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.zeros(4))
                pyps = pyex.piFluxSolver(-2 * currJH[i][0], 1, -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.ones(4) * np.pi)
                pyp0 = pyex.piFluxSolver(-2 * currJH[i][0], 1, -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.array([np.pi, np.pi, 0, 0]))
                pyzp = pyex.piFluxSolver(-2 * currJH[i][0], 1, -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.array([0, 0, np.pi, np.pi]))
            py0s.solvemeanfield()
            pyps.solvemeanfield()
            pyp0.solvemeanfield()
            pyzp.solvemeanfield()
            GS = np.array([py0s.GS(), pyps.GS(), pyp0.GS(), pyzp.GS()])
            a = np.argmin(GS)
            sendtemp2[i] = a
            if a == 0:
                sendtemp[i] = GS[a]
                sendtemp1[i] = py0s.occu_num()
            elif a == 1:
                sendtemp[i] = GS[a]
                sendtemp1[i] = pyps.occu_num()
            elif a == 2:
                sendtemp[i] = GS[a]
                sendtemp1[i] = pyp0.occu_num()
            else:
                sendtemp[i] = GS[a]
                sendtemp1[i] = pyzp.occu_num()
        except:
            sendtemp[i] = np.NaN
            sendtemp1[i] = np.NaN
            sendtemp2[i] = np.NaN

    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    if rank == 0:
        rectemp = rectemp.reshape((nK, nH))
        rectemp1 = rectemp1.reshape((nK, nH))
        rectemp2 = rectemp2.reshape((nK, nH))
        np.savetxt('Files/' + filename + '_GS.txt', rectemp)
        np.savetxt('Files/' + filename + '_N.txt', rectemp1)
        np.savetxt('Files/' + filename + '.txt', rectemp2)
        JP = np.linspace(JPm, JPmax, nK)
        h = np.linspace(hm, hmax, nH)
        graphColorMesh(JP, h, rectemp, 'Files/' + filename + '_GS')
        graphColorMesh(JP, h, rectemp1, 'Files/' + filename + '_N')
        graphColorMesh(JP, h, rectemp2, 'Files/' + filename)
def findPhaseMag111_ex(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename, Jxx=False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = np.mgrid[JPm:JPmax:1j * nK, hm:hmax:1j * nH].reshape(2, -1).T
    le = nK * nH
    nb = le / size

    leftK = int(rank * nb)
    rightK = int((rank + 1) * nb)
    currsizeK = rightK - leftK

    currJH = JH[leftK:rightK]

    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp1 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)

    rectemp = None
    rectemp1 = None
    rectemp2 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp1 = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)

    for i in range(currsizeK):
        try:
            if Jxx==True:
                py0s = pyex.piFluxSolver(1, 2 * currJH[i][0], -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.zeros(4))
                pyps = pyex.piFluxSolver(1, -2 * currJH[i][0], -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.ones(4) * np.pi)
            else:
                py0s = pyex.piFluxSolver(2 * currJH[i][0], 1, -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.zeros(4))
                pyps = pyex.piFluxSolver(-2 * currJH[i][0], 1, -2 * currJH[i][0], h=currJH[i][1], n=n, kappa=kappa,
                                         BZres=BZres, flux=np.ones(4) * np.pi)
            py0s.solvemeanfield()
            pyps.solvemeanfield()
            GS = np.array([py0s.GS(), pyps.GS()])
            a = np.argmin(GS)
            sendtemp2[i] = a
            if a == 0:
                sendtemp[i] = GS[a]
                sendtemp1[i] = py0s.occu_num()
            else:
                sendtemp[i] = GS[a]
                sendtemp1[i] = pyps.occu_num()
        except:
            sendtemp[i] = np.NaN
            sendtemp1[i] = np.NaN
            sendtemp2[i] = np.NaN

    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    if rank == 0:
        rectemp = rectemp.reshape((nK, nH))
        rectemp1 = rectemp1.reshape((nK, nH))
        rectemp2 = rectemp2.reshape((nK, nH))
        np.savetxt('Files/' + filename + '_GS.txt', rectemp)
        np.savetxt('Files/' + filename + '_N.txt', rectemp1)
        np.savetxt('Files/' + filename + '.txt', rectemp2)
        JP = np.linspace(JPm, JPmax, nK)
        h = np.linspace(hm, hmax, nH)
        graphColorMesh(JP, h, rectemp, 'Files/' + filename + '_GS')
        graphColorMesh(JP, h, rectemp1, 'Files/' + filename + '_N')
        graphColorMesh(JP, h, rectemp2, 'Files/' + filename)
def completeSpan_ex(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, flux, filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = np.mgrid[JPm:JPmax:1j*nK, hm:hmax:1j*nH].reshape(2,-1).T
    le = nK*nH
    nb = le/size

    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK


    currJH = JH[leftK:rightK]

    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp1 = np.zeros(currsizeK, dtype=np.float64)

    rectemp = None
    rectemp1 = None


    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp1 = np.zeros(le, dtype=np.float64)

    for i in range(currsizeK):
        py0s = pyex.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=flux)
        py0s.solvemeanfield()

        sendtemp[i] = py0s.GS()
        sendtemp1[i] = py0s.occu_num()


    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)

    if rank == 0:
        rectemp = rectemp.reshape((nK, nH))
        rectemp1 = rectemp1.reshape((nK, nH))

        ncfilename = 'Files/' + filename + '_full_info.nc'
        with nc.Dataset(ncfilename, "w") as dataset:
            dataset.createDimension("Jpm", nK)
            dataset.createDimension("h", nH)

            temp_var3 = dataset.createVariable("MFE", "f4", ("Jpm", "h"))
            temp_var3[:, :] = rectemp
            temp_var3.long_name = "Variational Energy"
            temp_var = dataset.createVariable("n", "f4", ("Jpm", "h"))
            temp_var[:, :] = rectemp1
            temp_var.long_name = "occupation number"


#endregion
