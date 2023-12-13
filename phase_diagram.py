import numpy as np
import matplotlib.pyplot as plt
from misc_helper import *
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import pyrochlore_general as pygen
import pyrochlore_dispersion_pi_old as pypiold
import pyrochlore_dispersion_pp00 as pypp00
import pyrochlore_dispersion_pi_gang_chen as pysung
import netCDF4 as nc
import warnings


def graphdispersion(Jxx, Jyy, Jzz, h, n, kappa, graphres, BZres, pi):
    if pi == 0:
        py0s = py0.zeroFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n)
    elif pi == 1:
        py0s = pypi.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n)
    else:
        py0s = pypp00.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n)
    py0s.solvemeanfield()
    print(py0s.lams, py0s.minLams, py0s.delta, py0s.qmin, py0s.condensed, py0s.MFE(), py0s.gap(), py0s.magnetization(), py0s.chi, py0s.chi0, py0s.xi)
    py0s.graph(False)
    return py0s.MF

# def testdispersion(Jxx, Jyy, Jzz, h, n, kappa, graphres, BZres):
#     py0s = pyarch.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n)
#     py0s.findLambda()
#     py0s.graph(False)
#     return py0s.MF

def sungdispersion(Jxx, Jyy, Jzz, h, n, kappa, graphres, BZres):
    Jpm = -(Jxx+Jyy)/4
    py0s = pysung.piFluxSolver(Jpm, Jzz, kappa=kappa, graphres=graphres, BZres=BZres)
    py0s.findLambda()
    py0s.graph(False)
    return py0s.MF

def generaldispersion(Jxx, Jyy, Jzz, h, n, kappa, graphres, BZres, flux):
    py0s = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    print(py0s.lams, py0s.minLams, py0s.delta, py0s.qmin, py0s.condensed, py0s.MFE(), py0s.gap(), py0s.magnetization(), py0s.chi, py0s.chi0, py0s.xi)
    py0s.graph(False)
    return py0s.MF

def graphedges(Jxx, Jyy, Jzz, h, n, kappa, rho, graphres, BZres, pi=False):
    if not pi:
        py0s = py0.zeroFluxSolver(Jxx, Jyy, Jzz,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
        py0s.solvemeanfield()
        py0s.graph_loweredge(False)
        py0s.graph_loweredge(True)
    else:
        py0s = pypi.piFluxSolver(Jxx, Jyy, Jzz, eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
        py0s.solvemeanfield()
        py0s.graph_loweredge(False)
        py0s.graph_upperedge(True)

def graphdispersion_old(JP,h, n, kappa, rho, graphres, BZres):
    py0s = pypiold.piFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
    py0s.findLambda()
    q2 = py0s.green_pi_branch(py0s.bigB)
    # temp = py0s.M_true(py0s.bigB)[:,0:4, 0:4] - np.conj(py0s.M_true(py0s.bigB)[:,4:8, 4:8])
    py0s.graph(True)

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

    plt.pcolormesh(X, Y, phases.T)

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


#region Phase for Anisotropy
def findPhase(nK, nE, res, filename):

    JP = np.linspace(0, 0.03, nK)
    kappaR = np.linspace(-1, 0, nE)
    kappa = (1+kappaR)/(1-kappaR)

    #
    # JP = [-gW]
    # kappa = [1]

    phases = np.zeros((nK,nE), dtype=int)
    totaltask = nK * nE
    increment = totaltask / 50

    print("Begin calculating Phase Diagram with Anisotropy")
    el = "==:==:=="
    count = 0
    for i in range (nK):
        for j in range (nE):
            start = time.time()
            count = count + 1
            print("Jpm is now " + str(JP[i]))
            print("Kappa is now " + str(kappa[j]))
            if JP[i] >= 0:
                py0s = py0.zeroFluxSolver(JP[i], eta=kappa[j], BZres=res, Jpmpm=0.02)
                print("Finding 0 Flux Lambda")
                py0s.findLambda()

                # zgaps[i, j, :] = np.array([py0s.gap(0), py0s.gap(1)])
                # zlambdas[i, j, :] = np.array([py0s.lamA, py0s.lamB])
                # # zGS[:,i] = np.array([py0s.GS(0), py0s.GS(1)]).T
                # zGS[i, j, :] = np.array([gK-py0s.lamA, gK*kappa[j] - py0s.lamB])
                #
                # print("Finished 0 Flux:")
                # print([zlambdas[i, j, :], zgaps[i, j, :], zGS[i, j, :]])

                phases[i][j] = phase0(py0s.lams, py0s.minLams, 0)
            else:
                pyps = pypiold.piFluxSolver(JP[i], kappa=kappa[j], BZres=res)
                print("Finding pi Flux Lambda")
                pyps.findLambda()

                #
                # pgaps[i, j, :] = np.array([pyps.gap(0), pyps.gap(1)])
                # plambdas[i, j, :] = np.array([pyps.lamA, pyps.lamB])
                # pGS[i, j, :] = np.array([gK-pyps.lamA, gK*kappa[j] - pyps.lamB])
                #
                # print("Finished pi Flux:")
                # print([plambdas[i, j, :], pgaps[i, j, :], pGS[i, j, :]])
                # print([pyps.lamA, pyps.lamB, pyps.minLams])
                phases[i][j] = phase0(pyps.lams, pyps.minLams, 1)
            end = time.time()
            el = (end - start)*(totaltask-count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            sys.stdout.flush()



    np.savetxt('Files/'+filename+'.txt', phases)
    graphPhase(filename)

#endregion

#region Phase for Magnetic Field


def generalJPSweep(JPm, JPmax, nK, h, n, BZres, kappa, fluxs, filename):

    JP = np.linspace(JPm, JPmax, nK)
    GS =  np.zeros((len(fluxs), nK))
    MFE = np.zeros((len(fluxs), nK))

    # for i in range (nH):
    for i in range (nK):
        print("Jpm is now " + str(JP[i]))
        for j in range(len(fluxs)):
            # if (fluxs[j] == np.zeros(4)).all():
            #     py0s = py0.zeroFluxSolver(-2 * JP[i], -2 * JP[i], 1, h=h, n=n, kappa=kappa, BZres=BZres)
            # else:
            py0s = pygen.piFluxSolver(-2*JP[i], -2*JP[i], 1, h = h, n=n, kappa=kappa, BZres=BZres, flux=fluxs[j])
            py0s.solvemeanfield()
            GS[j, i] = py0s.condensed
            MFE[j, i] = py0s.MFE()

    for i in range(len(fluxs)):
        plt.plot(JP, MFE[i], label=str(fluxs[i]))
    plt.legend()
    plt.xlabel(r'$J_\pm/J_{yy}$')
    plt.ylabel(r'$\omega/J_{yy}$')
    plt.savefig(filename+'.png')
    plt.clf()

def comparePi(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):

    JP = np.linspace(JPm, JPmax, nK)
    h = hm
    GS =  np.zeros((2, nK))
    MFE = np.zeros((2, nK))

    flux = np.ones(4)*np.pi
    # for i in range (nH):
    for i in range (nK):
        print("JP is now " + str(JP[i]))
        py0s = pypi.piFluxSolver(-2*JP[i], -2*JP[i], 1, h=h, n=n, kappa=kappa, BZres=BZres)
        py = pygen.piFluxSolver(-2*JP[i], -2*JP[i], 1, h = h, n=n, kappa=kappa, BZres=BZres, flux=flux)
        py0s.solvemeanfield()
        py.solvemeanfield()
        GS[0, i] = py0s.condensed
        MFE[0, i] = py0s.MFE()
        GS[1, i] = py.condensed
        MFE[1, i] = py.MFE()

    plt.plot(JP, MFE[0], label = "old")
    plt.plot(JP, MFE[1], label = "new")
    plt.legend()
    plt.xlabel(r'$h/J_{yy}$')
    plt.ylabel(r'$\omega/J_{yy}$')
    plt.savefig(filename+'.png')
    plt.clf()

    plt.plot(JP, MFE[1]-MFE[0])

    plt.legend()
    plt.savefig(filename+'_diff.png')
    plt.clf()


def compare0(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):

    JP = np.linspace(JPm, JPmax, nK)
    h = hm
    GS =  np.zeros((2, nK))
    MFE = np.zeros((2, nK))

    flux = np.zeros(4)
    # for i in range (nH):
    for i in range (nK):
        print("JP is now " + str(JP[i]))
        py0s = py0.zeroFluxSolver(-2*JP[i], -2*JP[i], 1, h=h, n=n, kappa=kappa, BZres=2*BZres)
        py = pygen.piFluxSolver(-2*JP[i], -2*JP[i], 1, h=h, n=n, kappa=kappa, BZres=BZres, flux=flux)
        py0s.solvemeanfield()
        py.solvemeanfield()
        GS[0, i] = py0s.condensed
        MFE[0, i] = py0s.MFE()
        GS[1, i] = py.condensed
        MFE[1, i] = py.MFE()
        print(MFE[0,i], MFE[1,i])

    plt.plot(JP, MFE[0], label = "old")
    plt.plot(JP, MFE[1], label = "new")
    plt.legend()
    plt.xlabel(r'$h/J_{yy}$')
    plt.ylabel(r'$\omega/J_{yy}$')
    plt.savefig(filename+'.png')
    plt.clf()

    plt.plot(JP, MFE[1]-MFE[0])

    plt.legend()
    plt.savefig(filename+'_diff.png')
    plt.clf()

def generalHSweep(JP, hm, hmax, nH, n, BZres, kappa, fluxs, filename):

    h = np.linspace(hm, hmax, nH)
    GS =  np.zeros((len(fluxs), nH))
    MFE = np.zeros((len(fluxs), nH))

    # for i in range (nH):
    for i in range (nH):
        print("h is now " + str(h[i]))
        for j in range(len(fluxs)):
            if (fluxs[j] == np.zeros(4)).all():
                py0s = py0.zeroFluxSolver(-2*JP, -2*JP, 1, h=h[i], n=n, kappa=kappa, BZres=BZres)
            else:
                py0s = pygen.piFluxSolver(-2*JP, -2*JP, 1, h = h[i], n=n, kappa=kappa, BZres=BZres, flux=fluxs[j])
            py0s.solvemeanfield()
            GS[j, i] = py0s.condensed
            MFE[j, i] = py0s.MFE()

    for i in range(len(fluxs)):
        plt.plot(h, MFE[i], label=str(fluxs[i]))

    plt.legend()
    plt.xlabel(r'$h/J_{yy}$')
    plt.ylabel(r'$\omega/J_{yy}$')
    plt.savefig(filename+'.png')
    plt.clf()

    for i in range(1, len(fluxs)):
        plt.plot(h, MFE[i]-MFE[0], label=str(fluxs[i]))

    plt.legend()
    plt.savefig(filename+'_diff.png')
    plt.clf()

def PhaseMagtestJP(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):

    JP = np.linspace(JPm, JPmax, nK)

    GS =  np.zeros(nK)
    MFE = np.zeros(nK)
    GSp =  np.zeros(nK)
    MFEp = np.zeros(nK)
    GSp0 =  np.zeros(nK)
    MFEp0 = np.zeros(nK)
    # for i in range (nH):
    for i in range (nK):
        print("Jpm is now " + str(JP[i]))
        py0s = py0.zeroFluxSolver(-2*JP[i], -2*JP[i], 1, h = hm, n=n, kappa=kappa, BZres=BZres)
        print("Finding 0 Flux Lambda")
        py0s.solvemeanfield()
        GS[i] = py0s.condensed
        MFE[i] = py0s.MFE()
        pyp = pypi.piFluxSolver(-2*JP[i], -2*JP[i], 1, h = hm, n=n, kappa=kappa, BZres=BZres)
        print("Finding pi Flux Lambda")
        pyp.solvemeanfield()
        GSp[i] = pyp.condensed
        MFEp[i] = pyp.MFE()
        pyp0 = pypp00.piFluxSolver(-2*JP[i], -2*JP[i], 1, h = hm, n=n, kappa=kappa, BZres=BZres)
        print("Finding pi pi 0 0 Lambda")
        pyp0.solvemeanfield()
        GSp0[i] = pyp0.condensed
        MFEp0[i] = pyp0.MFE()
        a = np.array([MFE[i], MFEp[i], MFEp0[i]])
        print(py0s.lams, pyp.lams, pyp0.lams)
        print(a)
        print("Phase is " + str(np.argmin(a)))


    # plt.plot(JP, gap, color='y')
    # plt.plot(JP, gapp, color='m')
    # plt.plot(JP, GS, color='r')
    # plt.plot(JP, GSp, color='b')
    plt.plot(JP, MFE, color='r')
    plt.plot(JP, MFEp, color='b')
    plt.plot(JP, MFEp0, color='g')

    # plt.plot(JP, condensed, color='y')
    # plt.plot(JP, condensed1, color='m')
    # plt.plot(JP, lamdiff, color='b')
    # plt.plot(JP, dev, color='black')
    plt.savefig(filename+'.png')
    # plt.show()

    plt.clf()
    # plt.show()

def MagJP(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):

    JP = np.linspace(JPm, JPmax, nK)

    Sx = np.zeros(nK)
    Sxp = np.zeros(nK)
    for i in range (nK):
        print("Jpm is now " + str(JP[i]))
        py0s = py0.zeroFluxSolver(-2*JP[i], -2*JP[i], 1, h = hm, n=n, kappa=kappa, BZres=BZres)
        print("Finding 0 Flux Lambda")
        py0s.solvemeanfield()
        Sx[i] = py0s.magnetization()
        print(py0s.gap())
        # pyp = pypi.piFluxSolver(-2*JP[i], -2*JP[i], 1, h = hm, n=n, kappa=kappa, BZres=BZres)
        # print("Finding pi Flux Lambda")
        # pyp.solvemeanfield()
        # Sxp[i] = pyp.magnetization()
        # print(Sx[i], Sxp[i])
    plt.plot(JP, Sx, color='r')
    # plt.plot(JP, Sxp, color='b')
    plt.savefig(filename+'.png')
    # plt.show()
    plt.clf()


def PhaseMagtestH(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):


    h = np.linspace(hm, hmax, nH)

    MFE = np.zeros(nH)
    MFEp = np.zeros(nH)
    MFEpp = np.zeros(nH)

    for i in range(nH):
        print("h is now " + str(h[i]))
        # print("h is now " + str(h[j]))
        py0s = py0.zeroFluxSolver(-2*JPm, -2*JPm, 1, h = h[i], n=n, kappa=kappa, BZres=BZres)
        print("Finding 0 Flux Lambda")
        py0s.solvemeanfield()
        MFE[i] = py0s.MFE()
        pyp = pypi.piFluxSolver(-2*JPm, -2*JPm, 1, h = h[i], n=n, kappa=kappa, BZres=BZres)
        print("Finding pi Flux Lambda")
        pyp.solvemeanfield()
        MFEp[i] = pyp.MFE()
        pyp0 = pypp00.piFluxSolver(-2*JPm, -2*JPm, 1, h = h[i], n=n, kappa=kappa, BZres=BZres)
        print("Finding pi Flux Lambda")
        pyp0.solvemeanfield()
        MFEpp[i] = pyp0.MFE()
        print(MFE[i], MFEp[i], MFEpp[i])

    plt.plot(h, MFE, color='r', label='0 Flux')
    plt.plot(h, MFEp, color='b', label=r'$\pi$ Flux')
    plt.plot(h, MFEpp, color='g', label=r'$\pi\pi 0 0$ Flux')
    plt.legend()
    plt.savefig(filename+'.png')
    plt.clf()

    plt.plot(h, MFEp - MFE, color='r', label=r'$\pi$ Flux')
    plt.plot(h, MFEpp - MFE, color='g', label=r'$\pi\pi 0 0$ Flux')
    plt.legend()
    plt.savefig(filename+'_diff.png')
    plt.clf()

def PhaseMagtestHGS(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):


    h = np.linspace(hm, hmax, nH)

    MFE = np.zeros(nH)
    MFEp = np.zeros(nH)
    MFEpp = np.zeros(nH)

    for i in range(nH):
        print("h is now " + str(h[i]))
        # print("h is now " + str(h[j]))
        py0s = py0.zeroFluxSolver(-2*JPm, -2*JPm, 1, h = h[i], n=n, kappa=kappa, BZres=BZres)
        print("Finding 0 Flux Lambda")
        py0s.solvemeanfield()
        MFE[i] = py0s.GS()
        pyp = pypi.piFluxSolver(-2*JPm, -2*JPm, 1, h = h[i], n=n, kappa=kappa, BZres=BZres)
        print("Finding pi Flux Lambda")
        pyp.solvemeanfield()
        MFEp[i] = pyp.GS()
        # pyp = pygang.piFluxSolver(JPm, h = h[i], n=n, kappa=kappa, BZres=BZres)
        # print("Finding pi Flux Lambda")
        # pyp.findLambda()
        # MFEpp[i] = pyp.GS()
        print(MFE[i], MFEp[i])

    plt.plot(h, MFE, color='r')
    plt.plot(h, MFEp, color='b')
    plt.savefig(filename+'.png')
    # plt.show()
    plt.clf()
    plt.plot(h, MFE-MFEp)
    plt.savefig(filename+'_diff.png')
    plt.clf()



def findPhaseMag_pi(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
    # totaltask = nK*nH
    # increment = totaltask/50
    # count = 0
    #

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nb = nK/size

    left = int(rank*nb)
    right = int((rank+1)*nb)
    currsize = right-left

    JP = np.linspace(JPm, JPmax, nK)
    currJP = JP[left:right]
    h = np.linspace(hm, hmax, nH)
    # phases = np.zeros((nK, nH), dtype=float)
    # gap = np.zeros((nK, nH), dtype=float)

    leng = len(np.concatenate((genBZ(BZres), symK)))

    sendtemp = np.zeros((currsize, nH), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, nH), dtype=np.float64)
    # sendtemp2 = np.zeros((currsize, nH, leng, 3), dtype=np.float64)

    rectemp = None
    rectemp1 = None
    # rectemp2 = None

    if rank == 0:
        rectemp = np.zeros((nK, nH), dtype=np.float64)
        rectemp1 = np.zeros((nK, nH), dtype=np.float64)
        # rectemp2 = np.zeros((nK, nH, leng, 3), dtype=np.float64)

    for i in range (currsize):
        for j in range (nH):
            # print(currJP[i], h[j])
            py0s = pypiold.piFluxSolver(currJP[i], h=h[j], n=n, kappa=kappa, BZres=BZres)

            py0s.findLambda()
            GSz = py0s.gap()

            py0s.findminLam()
            py0s.qvec()
            sendtemp[i,j] = py0s.condensed()[0]
            sendtemp1[i,j] = GSz
            # sendtemp2[i,j] = py0s.q

    sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0] * sendtemp1.shape[1], 0))
    # sendcounts2 = np.array(comm.gather(sendtemp2.shape[0] * sendtemp2.shape[1] * sendtemp2.shape[2] * sendtemp2.shape[3], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    # comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    if rank == 0:
        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_gap.txt', rectemp1)
        graphMagPhase(JP, h, rectemp1,'Files/' + filename + '_gap')
        plt.contourf(JP, h, rectemp.T)
        plt.xlabel(r'$J_\pm/J_{y}$')
        plt.ylabel(r'$h/J_{y}$')
        plt.savefig('Files/' + filename+'.png')
        plt.clf()
        # ncfilename = 'Files/' + filename + '_q_condensed.nc'
        # with nc.Dataset(ncfilename, "w") as dataset:
        #     # Create dimensions
        #     dataset.createDimension("Jpm", nK)
        #     dataset.createDimension("h", nH)
        #     dataset.createDimension("n", leng)
        #     dataset.createDimension("xyz", 3)

        #     temp_var = dataset.createVariable("q_condensed", "f4", ("Jpm", "h", "n", "xyz"))

        #     # Assign data to variables
        #     temp_var[:, :, :, :] = rectemp2

        #     # Add attributes
        #     temp_var.long_name = "Condensed Wave Vectors"


def findPhaseMag_zero(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
    # totaltask = nK*nH
    # increment = totaltask/50
    # count = 0
    #

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nb = nK/size

    left = int(rank*nb)
    right = int((rank+1)*nb)
    currsize = right-left

    JP = np.linspace(JPm, JPmax, nK)
    currJP = JP[left:right]
    h = np.linspace(hm, hmax, nH)
    # phases = np.zeros((nK, nH), dtype=float)
    # gap = np.zeros((nK, nH), dtype=float)

    leng = len(np.concatenate((genBZ(BZres), symK)))

    sendtemp = np.zeros((currsize, nH), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, nH), dtype=np.float64)
    # sendtemp2 = np.zeros((currsize, nH, leng, 3), dtype=np.float64)

    rectemp = None
    rectemp1 = None
    # rectemp2 = None

    if rank == 0:
        rectemp = np.zeros((nK, nH), dtype=np.float64)
        rectemp1 = np.zeros((nK, nH), dtype=np.float64)
        # rectemp2 = np.zeros((nK, nH, leng, 3), dtype=np.float64)

    for i in range (currsize):
        for j in range (nH):
            py0s = py0.zeroFluxSolver(currJP[i], h=h[j], n=n, kappa=kappa, BZres=BZres)

            py0s.findLambda()
            GSz = py0s.gap()

            py0s.findminLam()
            py0s.qvec()
            sendtemp[i,j] = py0s.condensed()[0]
            sendtemp1[i,j] = GSz
            # sendtemp2[i,j] = py0s.q

    sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0] * sendtemp1.shape[1], 0))
    # sendcounts2 = np.array(comm.gather(sendtemp2.shape[0] * sendtemp2.shape[1] * sendtemp2.shape[2] * sendtemp2.shape[3], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    # comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    if rank == 0:
        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_gap.txt', rectemp1)
        graphMagPhase(JP, h, rectemp1,'Files/' + filename + '_gap')
        plt.contourf(JP, h, rectemp.T)
        plt.xlabel(r'$J_\pm/J_{y}$')
        plt.ylabel(r'$h/J_{y}$')
        plt.savefig('Files/' + filename+'.png')
        plt.clf()
        # ncfilename = 'Files/' + filename + '_q_condensed.nc'
        # with nc.Dataset(ncfilename, "w") as dataset:
        #     # Create dimensions
        #     dataset.createDimension("Jpm", nK)
        #     dataset.createDimension("h", nH)
        #     dataset.createDimension("n", leng)
        #     dataset.createDimension("xyz", 3)

        #     temp_var = dataset.createVariable("q_condensed", "f4", ("Jpm", "h", "n", "xyz"))

        #     # Assign data to variables
        #     temp_var[:, :, :, :] = rectemp2

        #     # Add attributes
        #     temp_var.long_name = "Condensed Wave Vectors"


def findPhaseMag(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, flux, filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    JH = np.mgrid[JPm:JPmax:1j*nK, hm:hmax:1j*nH].reshape(2,-1).T
    le = len(JH)
    nb = le/size

    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK


    currJH = JH[leftK:rightK]


    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp1 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp4 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp5 = np.zeros((currsizeK,3), dtype=np.float64)


    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None
    rectemp5 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp1 = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.float64)
        rectemp4 = np.zeros(le, dtype=np.float64)
        rectemp5 = np.zeros((le, 3), dtype=np.float64)

    for i in range(currsizeK):
        start = time.time()
        py0s = pygen.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.zeros(4))
        pyps = pygen.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
        pyp0 = pygen.piFluxSolver(-2*currJH[i][0], -2*currJH[i][0], 1, h=currJH[i][1], n=n, kappa=kappa, BZres=BZres, flux=flux)

        py0s.solvemeanfield()
        pyps.solvemeanfield()
        pyp0.solvemeanfield()
        GS = np.array([py0s.MFE(), pyps.MFE(), pyp0.MFE()])
        a = np.argmin(GS)

        if a == 0:
            sendtemp1[i] = py0s.gap()
            sendtemp[i] = py0s.condensed
            sendtemp2[i] = GS[a]
            sendtemp3[i] = py0s.lams[0]
            sendtemp4[i] = py0s.magnetization()
            sendtemp5[i] = py0s.qmin.reshape(3,-1)
        elif a == 1:
            sendtemp1[i] = pyp0.gap()
            sendtemp[i] = pyp0.condensed + 5
            sendtemp2[i] = GS[a]
            sendtemp3[i] = pyp0.lams[0]
            sendtemp4[i] = pyp0.magnetization()
            sendtemp5[i] = py0s.qmin.reshape(3,-1)
        else:
            sendtemp1[i] = pyp0.gap()
            sendtemp[i] = pyp0.condensed + 10
            sendtemp2[i] = GS[a]
            sendtemp3[i] = pyp0.lams[0]
            sendtemp4[i] = pyp0.magnetization()
            sendtemp5[i] = py0s.qmin.reshape(3,-1)
        end = time.time()
        print("This iteration costs " + str(end - start))
#


    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0], 0))
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0], 0))
    sendcounts5 = np.array(comm.gather(sendtemp5.shape[0]*sendtemp5.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)

    # comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    if rank == 0:
        rectemp = rectemp.reshape((nK, nH))
        rectemp1 = rectemp1.reshape((nK, nH))
        rectemp2 = rectemp2.reshape((nK, nH))
        rectemp3 = rectemp3.reshape((nK, nH))
        rectemp4 = rectemp4.reshape((nK, nH))
        rectemp5 = rectemp5.reshape((nK, nH, 3))
        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_gap.txt', rectemp1)
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
            temp_var[:, :, :, :] = rectemp5
            # Add attributes
            temp_var.long_name = "Condensed Wave Vectors"


        JP = np.linspace(JPm, JPmax, nK)
        h = np.linspace(hm, hmax, nH)
        graphMagPhase(JP, h, rectemp, 'Files/' + filename)
        graphMagPhase(JP, h, rectemp1, 'Files/' + filename + '_gap')
        graphMagPhase(JP, h, rectemp2,'Files/' + filename + '_MFE')
        graphMagPhase(JP, h, rectemp3,'Files/' + filename + '_lam')
        graphMagPhase(JP, h, rectemp4,'Files/' + filename + '_mag')




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
    sendtemp1 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp4 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp5 = np.zeros(currsizeK, dtype=np.float64)

    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None
    rectemp5 = None

    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)
        rectemp1 = np.zeros(le, dtype=np.float64)
        rectemp2 = np.zeros(le, dtype=np.float64)
        rectemp3 = np.zeros(le, dtype=np.float64)
        rectemp4 = np.zeros(le, dtype=np.float64)
        rectemp5 = np.zeros(le, dtype=np.float64)

    for i in range (currsizeK):
        py0s = pygen.piFluxSolver(currJH[i][0], currJH[i][1], 1, kappa=kappa, BZres=BZres, flux=np.zeros(4))
        py0s.solvemeanfield()
        pyps = pygen.piFluxSolver(currJH[i][0], currJH[i][1], 1, kappa=kappa, BZres=BZres, flux=np.ones(4)*np.pi)
        pyps.solvemeanfield()
        GS = py0s.MFE()
        GSp = pyps.MFE()
        if GS < GSp:
            # py0s = py0.zeroFluxSolver(currJH[i][0], currJH[i][1], 1, kappa=kappa, BZres=BZres)
            # py0s.solvemeanfield(1e-4)
            sendtemp1[i] = py0s.gap()
            sendtemp[i] = py0s.condensed
            sendtemp2[i] = GS
            sendtemp3[i] = py0s.xi
            sendtemp4[i] = py0s.chi
            sendtemp5[i] = py0s.chi0
        else:
            # pyps = pypi.piFluxSolver(currJH[i][0], currJH[i][1], 1, kappa=kappa, BZres=BZres)
            # pyps.solvemeanfield(1e-4)
            sendtemp1[i] = pyps.gap()
            sendtemp[i] = pyps.condensed + 5
            sendtemp2[i] = GSp
            sendtemp3[i] = pyps.xi
            sendtemp4[i] = pyps.chi
            sendtemp5[i] = pyps.chi0

        end = time.time()
        # print(currJH[i], JPm, "This iteration costs " + str(end-start))


    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0], 0))
    sendcounts3 =  np.array(comm.gather(sendtemp3.shape[0], 0)) 
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0], 0))
    sendcounts5 =  np.array(comm.gather(sendtemp5.shape[0], 0)) 

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)

    if rank == 0:
        rectemp = rectemp.reshape((nK,nK))
        rectemp1 = rectemp1.reshape((nK,nK))
        rectemp2 = rectemp2.reshape((nK,nK))
        rectemp3 = rectemp3.reshape((nK,nK))
        rectemp4 = rectemp4.reshape((nK,nK))
        rectemp5 = rectemp5.reshape((nK,nK))

        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_gap.txt', rectemp1)
        np.savetxt('Files/' + filename + '_MFE.txt', rectemp2)
        np.savetxt('Files/' + filename + '_xi.txt', rectemp3)
        np.savetxt('Files/' + filename + '_chi.txt', rectemp4)
        np.savetxt('Files/' + filename + '_chi0.txt', rectemp5)

        JP = np.linspace(JPm, JPmax, nK)

        graphMagPhase(JP, JP, rectemp1,'Files/' + filename + '_gap')
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