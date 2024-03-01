import matplotlib.pyplot as plt
import netCDF4
import numpy as np

from misc_helper import *
import pyrochlore_gmft as pycon
import pyrochlore_exclusive_boson as pyex
import netCDF4 as nc
from mpi4py import MPI

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
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def plotLinefromnetCDF(field_direction, directory="Nia_Full_Scan/", Jpm=None, h=None, diff=False):
    if Jpm == None and h == None:
        return -1
    if (field_direction == h110).all():
        plotLine110(directory,Jpm, h, diff)
    elif (field_direction == h111).all():
        plotLine111(directory,Jpm, h, diff)
    else:
        plotLine100(directory,Jpm, h, diff)

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
            plt.plot(JPf, np.zeros(100))
            for i in range(3):
                MFEs[i] = readLineMFEh(ftoread[i+1], dex)
                plt.plot(JPf, MFEs[i]-zero)
            plt.legend(["0 flux", r"$\pi$ flux", r"$\pi\pi 0 0$", r"$ 00\pi\pi $"])
        else:
            MFEs = np.zeros((4, 100))
            for i in range(4):
                MFEs[i] = readLineMFEh(ftoread[i], dex)
                plt.plot(JPf, MFEs[i])
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
            plt.plot(hf, np.zeros(100))
            pi = readLineMFEh(ftoread[1], dex)
            plt.plot(hf, pi-zero)
            plt.legend(["0 flux", r"$\pi$ flux"])
        else:
            MFEs = np.zeros((2, 100))
            for i in range(2):
                MFEs[i] = readLineMFEh(ftoread[i], dex)
                plt.plot(JPf, MFEs[i])
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
            plt.plot(JPf, np.zeros(100))
            for i in range(3):
                MFEs[i] = readLineMFEh(ftoread[i+1], dex)
                plt.plot(JPf, MFEs[i]-zero)
            plt.legend(["0 flux", r"$\pi$ flux", r"$ 0\pi\pi 0$", r"$ \pi 00\pi $"])
        else:
            MFEs = np.zeros((4, 100))
            for i in range(4):
                MFEs[i] = readLineMFEh(ftoread[i], dex)
                plt.plot(JPf, MFEs[i])
            plt.legend(["0 flux", r"$\pi$ flux", r"$ 0\pi\pi 0$", r"$ \pi 00\pi $"])

#endregion


#region Phase for Magnetic Field

def findPhaseMag110(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
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
        # start = time.time()
        # print(currJH[i])
        py0s = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
        pyps = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
        pyp0 = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.array([np.pi,np.pi,0,0]))
        pyzp = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.array([0,0,np.pi,np.pi]))

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
            # sendtemp5[i] = np.array2string(pyp0.qmin)
        # print(sendtemp5[i], np.array2string(py0s.qmin),np.array2string(pyps.qmin),np.array2string(pyp0.qmin))
        # end = time.time()
        # print("This iteration costs " + str(end - start))
#


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

def findPhaseMag001(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
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
        # start = time.time()
        # print(currJH[i])
        py0s = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
        pyps = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
        pyp0 = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.array([np.pi,0,0,np.pi]))
        pyzp = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.array([0,np.pi,np.pi,0]))

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
            # sendtemp5[i] = np.array2string(pyp0.qmin)
        # print(sendtemp5[i], np.array2string(py0s.qmin),np.array2string(pyps.qmin),np.array2string(pyp0.qmin))
        # end = time.time()
        # print("This iteration costs " + str(end - start))
#


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

def findPhaseMag111(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
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
    # sendtemp5 = np.zeros(currsizeK, dtype='<U12')


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
        # rectemp5 = np.zeros(le, dtype='<U12')

    for i in range(currsizeK):
        # start = time.time()
        # print(currJH[i])
        py0s = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
        pyps = pycon.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)

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
#


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

def completeSpan(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, flux, filename):
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
    sendtemp5 = np.zeros((currsizeK,16, 3), dtype=np.float64)


    rectemp = None
    rectemp2 = None
    rectemp3 = None
    rectemp5 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.float64)
        rectemp5 = np.zeros((le, 16, 3), dtype=np.float64)

    for i in range(currsizeK):
        # start = time.time()
        # print(currJH[i])
        py0s = pycon.piFluxSolver(currJH[i][0], currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=flux)
        py0s.solvemeanfield()

        sendtemp[i] = py0s.condensed
        sendtemp2[i] = py0s.MFE()
        sendtemp3[i] = py0s.lams[0]
        temp = py0s.qminT
        leng = len(temp)
        sendtemp5[i,0:leng] = py0s.qminT


    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0], 0))
    sendcounts5 = np.array(comm.gather(sendtemp5.shape[0]*sendtemp5.shape[1]**sendtemp5.shape[2], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)


    if rank == 0:
        rectemp = rectemp.reshape((nK, nH))
        rectemp2 = rectemp2.reshape((nK, nH))
        rectemp3 = rectemp3.reshape((nK, nH))
        rectemp5 = rectemp5.reshape((nK, nH, 16, 3))

        ncfilename = 'Files/' + filename + '_full_info.nc'
        with nc.Dataset(ncfilename, "w") as dataset:
            dataset.createDimension("Jpm", nK)
            dataset.createDimension("h", nH)
            dataset.createDimension("index", 16)
            dataset.createDimension("xyz", 3)
            temp_var1 = dataset.createVariable("q_condensed", "f4", ("Jpm", "h", "index", "xyz"))
            temp_var1[:, :, :] = rectemp5
            temp_var1.long_name = "Condensed Wave Vectors"
            temp_var2 = dataset.createVariable("lams", "f4", ("Jpm", "h"))
            temp_var2[:, :] = rectemp3
            temp_var2.long_name = "lambda"
            temp_var3 = dataset.createVariable("MFE", "f4", ("Jpm", "h"))
            temp_var3[:, :] = rectemp2
            temp_var3.long_name = "Variational Energy"
            temp_var = dataset.createVariable("condensed", "f4", ("Jpm", "h"))
            temp_var[:, :] = rectemp
            temp_var.long_name = "isCondensed"

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

    JH = np.mgrid[JPm:JPmax:1j*nK, JP1m:JP1max:1j*nK].reshape(2,-1).T
    le = len(JH)
    nb = le/size

    leftK = int(rank * nb)
    rightK = int((rank + 1) * nb)
    currsizeK = rightK - leftK


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

    for i in range (currsizeK):
        py0s = pycon.piFluxSolver(currJH[i][0], currJH[i][1], 1, kappa=kappa, BZres=BZres, flux=np.zeros(4))
        py0s.solvemeanfield()
        pyps = pycon.piFluxSolver(currJH[i][0], currJH[i][1], 1, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
        pyps.solvemeanfield()
        GS = py0s.MFE()
        GSp = pyps.MFE()
        if GS < GSp:
            # py0s = py0.zeroFluxSolver(currJH[i][0], currJH[i][1], 1, kappa=kappa, BZres=BZres)
            # py0s.solvemeanfield(1e-4)
            sendtemp[i] = py0s.condensed
            sendtemp2[i] = GS
            sendtemp3[i] = py0s.xi
            sendtemp4[i] = py0s.chi
            sendtemp5[i] = py0s.chi0
        else:
            # pyps = pypi.piFluxSolver(currJH[i][0], currJH[i][1], 1, kappa=kappa, BZres=BZres)
            # pyps.solvemeanfield(1e-4)
            sendtemp[i] = pyps.condensed + 5
            sendtemp2[i] = GSp
            sendtemp3[i] = pyps.xi
            sendtemp4[i] = pyps.chi
            sendtemp5[i] = pyps.chi0

        end = time.time()
        # print(currJH[i], JPm, "This iteration costs " + str(end-start))


    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 =  np.array(comm.gather(sendtemp3.shape[0], 0))
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0], 0))
    sendcounts5 =  np.array(comm.gather(sendtemp5.shape[0], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)

    if rank == 0:
        rectemp = rectemp.reshape((nK,nK))
        rectemp2 = rectemp2.reshape((nK,nK))
        rectemp3 = rectemp3.reshape((nK,nK))
        rectemp4 = rectemp4.reshape((nK,nK))
        rectemp5 = rectemp5.reshape((nK,nK))

        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_MFE.txt', rectemp2)
        np.savetxt('Files/' + filename + '_xi.txt', rectemp3)
        np.savetxt('Files/' + filename + '_chi.txt', rectemp4)
        np.savetxt('Files/' + filename + '_chi0.txt', rectemp5)

        JP = np.linspace(JPm, JPmax, nK)

        plt.contourf(JP, JP, rectemp.T)
        plt.xlabel(r'$J_\pm/J_{y}$')
        plt.ylabel(r'$h/J_{y}$')
        plt.savefig('Files/' + filename+'.png')
        plt.clf()

        plt.contourf(JP, JP, rectemp2.T)
        plt.xlabel(r'$J_\pm/J_{y}$')
        plt.ylabel(r'$h/J_{y}$')
        plt.savefig('Files/' + filename+'_MFE.png')
        plt.clf()

        plt.contourf(JP, JP, rectemp3.T)
        plt.xlabel(r'$J_\pm/J_{y}$')
        plt.ylabel(r'$h/J_{y}$')
        plt.savefig('Files/' + filename+'_xi.png')
        plt.clf()

        plt.contourf(JP, JP, rectemp4.T)
        plt.xlabel(r'$J_\pm/J_{y}$')
        plt.ylabel(r'$h/J_{y}$')
        plt.savefig('Files/' + filename+'_chi.png')
        plt.clf()

        plt.contourf(JP, JP, rectemp5.T)
        plt.xlabel(r'$J_\pm/J_{y}$')
        plt.ylabel(r'$h/J_{y}$')
        plt.savefig('Files/' + filename+'_chi0.png')
        plt.clf()

#endregion

#region Phase for Magnetic Field - Exclusive Boson
def findPhaseMag110_ex(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
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
            py0s = pyex.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
            pyps = pyex.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
            pyp0 = pyex.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.array([np.pi,np.pi,0,0]))
            pyzp = pyex.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.array([0,0,np.pi,np.pi]))

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


def findPhaseMag001_ex(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
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
            py0s = pyex.piFluxSolver(-2 * currJH[i][0], -2 * currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa,
                                     BZres=BZres, flux=np.zeros(4))
            pyps = pyex.piFluxSolver(-2 * currJH[i][0], -2 * currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa,
                                     BZres=BZres, flux=np.ones(4) * np.pi)
            pyp0 = pyex.piFluxSolver(-2 * currJH[i][0], -2 * currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa,
                                     BZres=BZres, flux=np.array([0 ,np.pi, np.pi, 0]))
            pyzp = pyex.piFluxSolver(-2 * currJH[i][0], -2 * currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa,
                                     BZres=BZres, flux=np.array([np.pi, 0, 0, np.pi]))

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


def findPhaseMag111_ex(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
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
            py0s = pyex.piFluxSolver(-2 * currJH[i][0], -2 * currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa,
                                     BZres=BZres, flux=np.zeros(4))
            pyps = pyex.piFluxSolver(-2 * currJH[i][0], -2 * currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa,
                                     BZres=BZres, flux=np.ones(4) * np.pi)

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

#endregion