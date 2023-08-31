import numpy as np
import matplotlib.pyplot as plt
from misc_helper import *
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi

# JP, zgaps, zlambdas, zGS, pgaps, plambdas, pGS = np.loadtxt("test2.txt", delimiter=' ')
#
# plt.plot(JP, 0.5-zlambdas, JP, 0.5-plambdas)
# plt.legend(["Zero Flux", "Pi Flux"])
# plt.ylabel(r'$GS$')
# plt.xlabel(r'$J_{\pm}/J_{zz}$')
# plt.show()




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
                py0s = py0.zeroFluxSolver(JP[i], eta=kappa[j], BZres=res)
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
                pyps = pypi.piFluxSolver(JP[i], kappa=kappa[j], BZres=res)
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
    dev = np.zeros(nK)
    E111 = np.zeros(nK)
    E000 = np.zeros(nK)
    condensed = np.zeros(nK)
    # for i in range (nH):
    for i in range (nK):
        print("Jpm is now " + str(JP[i]))
        # print("h is now " + str(h[j]))
        if JP[i] >= 0:
            py0s = py0.zeroFluxSolver(JP[i], h = 0, n=n, kappa=kappa, BZres=BZres)
            print("Finding 0 Flux Lambda")
            # phases[i][j] = py0s.phase_test()
            py0s.findLambda()
            # py0s.findminLam()
            gap[i] = py0s.gap()
            # condensed[i] = py0s.condensed()[0]
            # dev[i] = py0s.rho_dev()

        else:
            py0s = pypi.piFluxSolver(JP[i], h = 0, n=n, kappa=kappa, BZres=BZres)
            print("Finding pi Flux Lambda")
            # phases[i][j] = py0s.phase_test()
            py0s.findLambda()
            # py0s.findminLam()
            gap[i] = py0s.gap()
            # condensed[i] = py0s.condensed()[0]
            # dev[i] = py0s.rho_dev()
    plt.plot(JP, gap, color='b')
    # plt.plot(JP, condensed, color='b')
    # plt.plot(JP, dev, color='black')
    plt.show()


def PhaseMagtestH(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):

    h = np.linspace(hm, hmax, nH)

    gap = np.zeros(nH)
    E111 = np.zeros(nH)
    E000 = np.zeros(nH)
    # for i in range (nH):
    for i in range (nH):
        print("h is now " + str(h[i]))
        py0s = py0.zeroFluxSolver(0, h = h[i], n=n, kappa=kappa, BZres=BZres)
        print("Finding 0 Flux Lambda")
        # phases[i][j] = py0s.phase_test()
        py0s.findLambda()
        # print(py0s.lams, py0s.minLams)
        # try:
        #     phases[i, j] = py0s.gap()
        # except:
        #     phases[i, j] = 0
        # try:
        #     if py0s.gap() < 1e-4:
        #         phases[i, j] = 0
        #     else:
        #         phases[i, j] = 1
        # except:
        #     phases[i, j] = 0
        gap[i] = py0s.gap()
        E111[i] = np.min(py0s.E_single(np.pi*np.array([1,1,1])))
        E000[i] = np.min(py0s.E_single(np.array([0,0,0])))
        print([gap[i], E111[i], E000[i]])
    plt.plot(h, gap, color='b')
    plt.plot(h, E111, color='g')
    plt.plot(h, E000, color='r')
    plt.show()



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
    phases = np.zeros((nK, nH), dtype=float)
    gap = np.zeros((nK, nH), dtype=float)

    sendtemp = np.zeros((currsize, nH), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, nH), dtype=np.float64)
    sendtemp2 = np.zeros((currsize, nH, 3), dtype=np.float64)

    rectemp = None
    rectemp1 = None
    rectemp2 = None

    if rank == 0:
        rectemp = np.zeros((currsize, nH), dtype=np.float64)
        rectemp1 = np.zeros((currsize, nH), dtype=np.float64)
        rectemp2 = np.zeros((currsize, nH, 3), dtype=np.float64)

    for i in range (currsize):
        for j in range (nH):
            # start = time.time()
            # count = count + 1
            if currJP[i] >= 0:
                py0s = py0.zeroFluxSolver(currJP[i], h=h[j], n=n, kappa=kappa, BZres=BZres)
            else:
                py0s = pypi.piFluxSolver(currJP[i], h=h[j], n=n, kappa=kappa, BZres=BZres)

            py0s.findLambda()
            py0s.findminLam()
            sendtemp[i,j] = py0s.condensed()[0]

            if phases[i,j]:
                sendtemp2[i,j] = py0s.gapwhere()
            else:
                sendtemp2[i,j] = -1000*np.ones(3)

            sendtemp1[i,j] = py0s.gap()
            # end = time.time()
            # el = (end - start) * (totaltask - count)
            # el = telltime(el)
            # sys.stdout.write('\r')
            # sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count / increment) + '-' * (50 - int(count / increment)), count / totaltask * 100, el))
            # sys.stdout.flush()
    sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0] * sendtemp1.shape[1], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0] * sendtemp2.shape[1] * sendtemp2.shape[2], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    if rank == 0:
        np.savetxt('Files/' + filename+'.txt', rectemp)
        np.savetxt('Files/' + filename + '_gap.txt', rectemp1)
        np.savetxt('Files/' + filename + '_q_condensed.txt', rectemp2.reshape(3,-1))
        graphMagPhase(JP, h, rectemp1,'Files/' + filename + '_gap')

#endregion