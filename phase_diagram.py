import numpy as np
import matplotlib.pyplot as plt
from misc_helper import *
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import pyrochlore_dispersion_pi_gang_chen as pygang
import pyrochlore_dispersion_pi_old as pypiold
import netCDF4 as nc
import warnings

# JP, zgaps, zlambdas, zGS, pgaps, plambdas, pGS = np.loadtxt("test2.txt", delimiter=' ')
#
# plt.plot(JP, 0.5-zlambdas, JP, 0.5-plambdas)
# plt.legend(["Zero Flux", "Pi Flux"])
# plt.ylabel(r'$GS$')
# plt.xlabel(r'$J_{\pm}/J_{zz}$')
# plt.show()


def graphdispersion(Jxx, Jyy, Jzz, h, n, kappa, rho, graphres, BZres, pi):
    if not pi:
        py0s = py0.zeroFluxSolver(Jxx, Jyy, Jzz,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
    else:
        py0s = pypi.piFluxSolver(Jxx, Jyy, Jzz,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
    py0s.solvemeanfield(1e-3)
    print(py0s.lams, py0s.minLams, py0s.delta, py0s.qmin, py0s.condensed, py0s.chi, py0s.chi0, py0s.xi)
    py0s.graph(False)
    return 0

def graphedges(JP,h, n, kappa, rho, graphres, BZres, old=False):
    if JP >= 0:
        py0s = py0.zeroFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
        py0s.findminLam()
        py0s.findLambda()
        py0s.graph_loweredge(False)
        py0s.graph_loweredge(True)
    elif JP < 0 and not old:
        py0s = pypi.piFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
        py0s.findLambda()
        py0s.graph_loweredge(False)
        py0s.graph_upperedge(True)
    else:
        py0s = pypiold.piFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
        py0s.findLambda()
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

    # plt.imshow(bigphase, cmap='gray', vmin=-3, vmax=3, interpolation='bilinear', extent=[-0.1, 0.1, -1, 1], aspect='auto')

    # cs = plt.contourf(X, Y, phases, levels=[0, 0.13,10000], colors=['#43AC63', '#B5E8C4'])
    plt.contourf(X, Y, phases.T, levels=100)
    # plt.colorbar()

    plt.xlabel(r'$J_\pm/J_{y}$')
    plt.ylabel(r'$h/J_{y}$')
    plt.savefig(filename +'.png')
    plt.clf()

    plt.contourf(X, Y, phases.T, levels=[0, 0.05,10000], colors=['#43AC63', '#B5E8C4'])
    # plt.colorbar()

    plt.xlabel(r'$J_\pm/J_{y}$')
    plt.ylabel(r'$h/J_{y}$')
    plt.savefig(filename+'_split.png')
    plt.clf()


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

def PhaseMagtestJP(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):

    JP = np.linspace(JPm, JPmax, nK)

    gap = np.zeros(nK)
    GS =  np.zeros(nK)
    MFE = np.zeros(nK)
    gapp = np.zeros(nK)
    GSp =  np.zeros(nK)
    MFEp = np.zeros(nK)
    E000 = np.zeros(nK)
    condensed = np.zeros(nK)
    condensed1 = np.zeros(nK)
    lamdiff = np.zeros(nK)
    # for i in range (nH):
    for i in range (nK):
        print("Jpm is now " + str(JP[i]))
        # print("h is now " + str(h[j]))
        py0s = py0.zeroFluxSolver(-2*JP[i], -2*JP[i], 1, h = hm, n=n, kappa=kappa, BZres=BZres)
        print("Finding 0 Flux Lambda")
        # phases[i][j] = py0s.phase_test()
        py0s.solvemeanfield()
        # print(py0s.lams, py0s.minLams)
        # gap[i] = py0s.gap()
        # GS[i] = py0s.condensed()[0]
        MFE[i] = py0s.MFE()
        # print(MFE[i])
        pyp = pypi.piFluxSolver(-2*JP[i], -2*JP[i], 1, h = hm, n=n, kappa=kappa, BZres=BZres)
        print("Finding pi Flux Lambda")
        # phases[i][j] = py0s.phase_test()
        pyp.solvemeanfield()
        # gapp[i] = pyp.gap()
        # GSp[i] = pyp.condensed()[0]
        MFEp[i] = pyp.MFE()
        print(MFE[i], MFEp[i])
        # lamdiff[i] = abs(pyp.minLams-pyp.lams)[0]
        # print(pyp.minLams, pyp.lams, lamdiff[i])
        # condensed[i] = pyp.condensed()[0]
        # condensed1[i] = 1 if gapp[i] < 0.06 else 0

        # condensed[i] = py0s.condensed()[0]
        # dev[i] = py0s.rho_dev()
    # plt.plot(JP, gap, color='y')
    # plt.plot(JP, gapp, color='m')
    # plt.plot(JP, GS, color='r')
    # plt.plot(JP, GSp, color='b')
    plt.plot(JP, MFE, color='r')
    plt.plot(JP, MFEp, color='b')
    # plt.plot(JP, condensed, color='y')
    # plt.plot(JP, condensed1, color='m')
    # plt.plot(JP, lamdiff, color='b')
    # plt.plot(JP, dev, color='black')
    plt.savefig(filename)
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
        # pyp = pypi.piFluxSolver(-2*JP[i], -2*JP[i], 1, h = hm, n=n, kappa=kappa, BZres=BZres)
        # print("Finding pi Flux Lambda")
        # pyp.solvemeanfield()
        # Sxp[i] = pyp.magnetization()
        # print(Sx[i], Sxp[i])
    plt.plot(JP, Sx, color='r')
    # plt.plot(JP, Sxp, color='b')
    plt.savefig(filename)
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


def findPhaseMag(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
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
            py0s = py0.zeroFluxSolver(-2*currJP[i], -2*currJP[i], 1, h=h[j], n=n, kappa=kappa, BZres=BZres)
            pyps = pypi.piFluxSolver(-2*currJP[i], -2*currJP[i], 1, h=h[j], n=n, kappa=kappa, BZres=BZres)

            py0s.solvemeanfield()
            pyps.solvemeanfield()
            GSz = py0s.MFE()
            GSp = pyps.MFE()

            if GSz < GSp:
                # py0s.qvec()
                # print(pyps.minLams)
                # warnings.filterwarnings("error")
                sendtemp1[i,j] = py0s.gap()
                # print(py0s.lams, py0s.minLams)
                sendtemp[i,j] = py0s.condensed
                # print(sendtemp[i,j])
                # sendtemp2[i,j] = py0s.q
            else:
                # print(pyps.minLams)
                # pyps.qvec()
                # warnings.filterwarnings("error")
                sendtemp1[i,j] = pyps.gap()
                # print(pyps.lams, pyps.minLams)
                sendtemp[i,j] = pyps.condensed + 5
                # print(sendtemp[i, j])
                # sendtemp2[i,j] = pyps.q

            # print(sendtemp[i,j])
# 


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



def findXYZPhase(JPm, JPmax, nK, BZres, kappa, filename):
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
    # phases = np.zeros((nK, nH), dtype=float)
    # gap = np.zeros((nK, nH), dtype=float)

    leng = len(np.concatenate((genBZ(BZres), symK)))

    sendtemp = np.zeros((currsize, nK), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, nK), dtype=np.float64)
    sendtemp2 = np.zeros((currsize, nK), dtype=np.float64)

    rectemp = None
    rectemp1 = None
    rectemp2 = None

    if rank == 0:
        rectemp = np.zeros((nK, nK), dtype=np.float64)
        rectemp1 = np.zeros((nK, nK), dtype=np.float64)
        rectemp2 = np.zeros((nK, nK), dtype=np.float64)

    for i in range (currsize):
        for j in range (nK):
            py0s = py0.zeroFluxSolver(currJP[i], JP[j], 1, kappa=kappa, BZres=BZres)
            pyps = pypi.piFluxSolver(currJP[i], JP[j], 1, kappa=kappa, BZres=BZres)
            py0s.solvemeanfield(1e-4)
            pyps.solvemeanfield(1e-4)
            GSz = py0s.MFE()
            GSp = pyps.MFE()
            if GSz < GSp:
                # py0s.qvec()
                # print(pyps.minLams)
                # warnings.filterwarnings("error")
                sendtemp1[i,j] = py0s.gap()
                # print(py0s.lams, py0s.minLams)
                sendtemp[i,j] = py0s.condensed
                sendtemp2[i,j] = py0s.xi
                # print(sendtemp[i,j])
                # sendtemp2[i,j] = py0s.q
            else:
                # print(pyps.minLams)
                # pyps.qvec()
                # warnings.filterwarnings("error")
                sendtemp1[i,j] = pyps.gap()
                # print(pyps.lams, pyps.minLams)
                sendtemp[i,j] = pyps.condensed + 5
                sendtemp2[i,j] = pyps.xi
                # print(sendtemp[i, j])
                # sendtemp2[i,j] = pyps.q
            print(sendtemp[i,j])

            # print(sendtemp[i,j])
# 


    sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0] * sendtemp1.shape[1], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0] * sendtemp2.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    if rank == 0:
        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_gap.txt', rectemp1)
        np.savetxt('Files/' + filename + '_xi.txt', rectemp2)
        graphMagPhase(JP, JP, rectemp1,'Files/' + filename + '_gap')
        plt.contourf(JP, JP, rectemp.T)
        plt.xlabel(r'$J_\pm/J_{y}$')
        plt.ylabel(r'$h/J_{y}$')
        plt.savefig('Files/' + filename+'.png')
        plt.clf()

        plt.contourf(JP, JP, rectemp2.T)
        plt.xlabel(r'$J_\pm/J_{y}$')
        plt.ylabel(r'$h/J_{y}$')
        plt.savefig('Files/' + filename+'_xi.png')
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


#endregion