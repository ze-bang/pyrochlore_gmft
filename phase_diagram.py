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


def graphMagPhase(filename, jpmax, hmax):
    phases = np.loadtxt('Files/' + filename + '.txt', delimiter=' ').T


    JP = np.linspace(0, jpmax, phases.shape[1])
    h = np.linspace(0, hmax, phases.shape[0])

    X,Y = np.meshgrid(JP, h)

    # plt.imshow(bigphase, cmap='gray', vmin=-3, vmax=3, interpolation='bilinear', extent=[-0.1, 0.1, -1, 1], aspect='auto')

    # cs = plt.contourf(X, Y, phases, levels=[0, 0.13,10000], colors=['#43AC63', '#B5E8C4'])
    cs = plt.contourf(X, Y, phases, levels=100)
    # plt.colorbar()

    plt.xlabel(r'$J_\pm/J_{y}$')
    plt.ylabel(r'$h/J_{y}$')
    plt.savefig('Files/'+filename+'phase.png')


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
                pyps = pypi.piFluxSolver(JP[i], kappa=kappa[j], BZres=res, lam=abs(JP[i])*2)
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
            py0s.findminLam()
            gap[i] = py0s.gap()
            condensed[i] = py0s.condensed()[0]
            # dev[i] = py0s.rho_dev()
    plt.plot(JP, gap, color='b')
    plt.plot(JP, condensed, color='b')
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
    totaltask = nK*nH
    increment = totaltask/50
    count = 0

    JP = np.linspace(JPm, JPmax, nK)
    h = np.linspace(hm, hmax, nH)
    phases = np.zeros((nK, nH), dtype=float)

    con = np.array([0,0,0])

    for i in range (nK):
        for j in range (nH):
            start = time.time()
            count = count + 1
            if JP[0] >= 0:
                py0s = py0.zeroFluxSolver(JP[i], h = h[j], n=n, kappa=kappa, BZres=BZres)
                # phases[i][j] = py0s.phase_test()
                py0s.findLambda()
                py0s.findminLam()
                phases[i,j] = py0s.condensed()[0]
                temp = py0s.gapwhere()
                tempb = False
                if phases[i,j]:
                    for k in range(len(con)):
                        if (con[k] == temp).all():
                            phases[i,j] = k+10
                            tempb = True
                if tempb:
                    np.append(con, temp)


            end = time.time()
            el = (end - start) * (totaltask - count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count / increment) + '-' * (50 - int(count / increment)), count / totaltask * 100, el))
            sys.stdout.flush()
    np.savetxt('Files/' + filename+'.txt', phases)
    graphPhase(filename)

#endregion