import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
import math

def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

def phase0(lams, minLams, pi):
    lamA, lamB = lams
    if np.all(lams < minLams):
        print("AFM Phase")
        return 0
    elif lamA < minLams[0]:
        print("PMu Phase")
        return 1+pi
    elif lamB < minLams[1]:
        print("PMu Phase")
        return -(1+pi)
    else:
        print("QSL Phase")
        print(3*(-1)**pi)
        return 3*(-1)**pi


def phaseMag(lams, minLams, pi):
    lamA, lamB = lams
    if np.all(lams < minLams):
        print("AFM Phase")
        return 0
    elif lamA < minLams[0]:
        print("PMu Phase")
        return 1+pi
    elif lamB < minLams[1]:
        print("PMu Phase")
        return -(1+pi)
    else:
        print("QSL Phase")
        print(3*(-1)**pi)
        return 3*(-1)**pi

def findPhase(nK, nE, res, filename):

    JP = np.linspace(0, 0.1, nK)
    kappaR = np.linspace(-1, 0, nE)
    kappa = (1+kappaR)/(1-kappaR)

    #
    # JP = [-0.3]
    # kappa = [1]

    phases = np.zeros((nK,nE), dtype=int)



    for i in range (nK):
        for j in range (nE):
            print("Jpm is now " + str(JP[i]))
            print("Kappa is now " + str(kappa[j]))
            if JP[i] >= 0:
                py0s = py0.zeroFluxSolver(JP[i], eta=kappa[j], res=res)
                py0s.setupALL()
                print("Finding 0 Flux Lambda")
                py0s.findLambda()

                # zgaps[i, j, :] = np.array([py0s.gap(0), py0s.gap(1)])
                # zlambdas[i, j, :] = np.array([py0s.lamA, py0s.lamB])
                # # zGS[:,i] = np.array([py0s.GS(0), py0s.GS(1)]).T
                # zGS[i, j, :] = np.array([0.5-py0s.lamA, 0.5*kappa[j] - py0s.lamB])
                #
                # print("Finished 0 Flux:")
                # print([zlambdas[i, j, :], zgaps[i, j, :], zGS[i, j, :]])

                phases[i][j] = phase0(py0s.lams, py0s.minLams, 0)
            else:
                pyps = pypi.piFluxSolver(JP[i], kappa=kappa[j], res=res, lam=abs(JP[i])*2)
                pyps.setupALL()
                print("Finding pi Flux Lambda")
                pyps.findLambda()

                #
                # pgaps[i, j, :] = np.array([pyps.gap(0), pyps.gap(1)])
                # plambdas[i, j, :] = np.array([pyps.lamA, pyps.lamB])
                # pGS[i, j, :] = np.array([0.5-pyps.lamA, 0.5*kappa[j] - pyps.lamB])
                #
                # print("Finished pi Flux:")
                # print([plambdas[i, j, :], pgaps[i, j, :], pGS[i, j, :]])
                # print([pyps.lamA, pyps.lamB, pyps.minLams])
                phases[i][j] = phase0(pyps.lams, pyps.minLams, 1)

    # np.savetxt("JP_11.txt", JP)
    # np.savetxt("zlambdas_11.txt", zlambdas[:,:,0])
    # np.savetxt("zlambdas_21.txt", zlambdas[:,:,1])
    # np.savetxt("zgaps_11.txt", zgaps[:,:,0])
    # np.savetxt("zgaps_21.txt", zgaps[:,:,1])
    # np.savetxt("zGS_11.txt", zGS[:,:,0])
    # np.savetxt("zGS_21.txt", zGS[:,:,1])
    # np.savetxt("plambdas_11.txt", plambdas[:,:,0])
    # np.savetxt("plambdas_21.txt", plambdas[:,:,1])
    # np.savetxt("pgaps_11.txt", pgaps[:,:,0])
    # np.savetxt("pgaps_21.txt", pgaps[:,:,1])
    # np.savetxt("pGS_11.txt", pGS[:,:,0])
    # np.savetxt("pGS_21.txt", pGS[:,:,1])

    np.savetxt(filename, phases)


def edges(D, E, tol):
    D = D.transpose()
    mindex = np.zeros(D.shape[0])
    maxdex = np.zeros(D.shape[0])
    print(D.shape)
    for i in range(D.shape[0]):
        minfound = False
        maxfound = False
        for j in range(D.shape[1]):
            if D[i][j] >= tol and not minfound:
                mindex[i] = E[j]
                minfound = True
            if D[i][j] <= tol and minfound and (not maxfound):
                maxdex[i] = E[j]
                maxfound = True
    return [mindex, maxdex]

def spinon_continuum_zero(nE, nK, Jpm):
    py0s = py0.zeroFluxSolver(Jpm, res=nK)
    py0s.setupALL()
    py0s.findLambda()
    py0s.calDispersion()

    e = np.linspace(py0s.gap(0), py0s.EMAX(0)*2.1, nE)
    kk = np.concatenate((np.linspace(-0.5, 0, nK), np.linspace(0, 0.3, nK), np.linspace(0.3, 0.5, nK), np.linspace(0.5,1, nK), np.linspace(1, 1.4, nK), np.linspace(1.4, 1.7, nK), np.linspace(1.7, 1.85, nK)))
    d1 = graph_spin_cont_zero(py0s, e, np.concatenate((py0s.GammaX, py0s.XW, py0s.WK, py0s.KGamma, py0s.GammaL, py0s.LU, py0s.UW)), 5e-2)
    np.savetxt("Files/spin_cont_test.txt", d1)

    # d1 = np.loadtxt("Files/spin_cont_test.txt")

    X,Y = np.meshgrid(kk, e)
    plt.contourf(X,Y, d1, levels=100)
    plt.ylabel(r'$\omega/J_{zz}$')
    # plt.axvline(x=-0.5, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=0, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=0.3, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=0.5, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=1, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=1.4, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=1.7, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=1.85, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [-0.5, 0, 0.3, 0.5, 1, 1.4, 1.7, 1.85]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)
    # dex = edges(d1, e, 5e-2)
    # plt.plot(kk, dex[0], 'b', kk, dex[1], 'b')

    plt.show()

def spinon_continuum_pi(nE, nK, Jpm):

    py0s = pypi.piFluxSolver(Jpm, res=nK)
    py0s.setupALL()
    py0s.findLambda()
    py0s.calAllDispersion()

    e = np.linspace(py0s.gap(0), py0s.EMAX(0)*2.1, nE)
    kk = np.concatenate((np.linspace(-0.5, 0, nK), np.linspace(0, 0.3, nK), np.linspace(0.3, 0.5, nK), np.linspace(0.5,1, nK), np.linspace(1, 1.4, nK), np.linspace(1.4, 1.7, nK), np.linspace(1.7, 1.85, nK)))
    # d1 = graph_spin_cont_pi(py0s, e, np.concatenate((py0s.GammaX, py0s.XW, py0s.WK, py0s.KGamma, py0s.GammaL, py0s.LU, py0s.UW)), 1e-4)
    d1 = graph_spin_cont_pi(py0s, e, py0s.GammaX, 1e-4)
    kk = np.linspace(-0.5, 0, nK)
    np.savetxt("Files/spin_cont_test_pi.txt", d1)

    # d1 = np.loadtxt("Files/spin_cont_test.txt")

    X,Y = np.meshgrid(kk, e)
    plt.contourf(X,Y, d1, levels=100)
    plt.ylabel(r'$\omega/J_{zz}$')
    # plt.axvline(x=-0.5, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=0, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=0.3, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=0.5, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=1, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=1.4, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=1.7, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=1.85, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [-0.5, 0, 0.3, 0.5, 1, 1.4, 1.7, 1.85]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)
    # dex = edges(d1, e, 5e-2)
    # plt.plot(kk, dex[0], 'b', kk, dex[1], 'b')

    plt.show()

def spinon_continuum(nE, nK, Jpm):
    if Jpm >= 0:
        spinon_continuum_zero(nE, nK, Jpm)
    else:
        spinon_continuum_pi(nE, nK, Jpm)

def findPhaseMag(JPm, JPmax, nK, hm, hmax, nH, n, kappa, filename):
    n = n / magnitude(n)
    print(n)

    JP = np.linspace(JPm, JPmax, nK)
    h = np.linspace(hm, hmax, nH)
    phases = np.zeros((nK,nH), dtype=int)


    for i in range (nK):
        for j in range (nH):
            print("Jpm is now " + str(JP[i]))
            print("h is now " + str(h[j]))
            if JP[i] >= 0:
                py0s = py0.zeroFluxSolver(JP[i], h= h[j], n=n, kappa=kappa, res=10)
                py0s.setupALL()
                print(py0s.minLams)
                print("Finding 0 Flux Lambda")
                py0s.findLambda()
                print([py0s.lams, py0s.minLams])
                phases[i][j] = phase0(py0s.lams, py0s.minLams, 0)
            else:
                pyps = pypi.piFluxSolver(JP[i], h= h[j], n=n, kappa=kappa, res=10)
                pyps.setupALL()
                print("Finding pi Flux Lambda")
                pyps.findLambda()
                print([pyps.lams, pyps.minLams])
                phases[i][j] = phase0(pyps.lams, pyps.minLams, 1)


    np.savetxt(filename, phases)

def graphdispersion(JP, kappa, rho, res):
    if JP >= 0:
        py0s = py0.zeroFluxSolver(JP, res=res)
        py0s.setupALL()
        py0s.findLambda()
        # print(py0s.lams)
        py0s.graph(0, False)
        py0s.graph(1, True)
    elif JP < 0:
        py0s = pypi.piFluxSolver(JP, res=res)
        py0s.setupALL()
        py0s.findLambda()
        py0s.graph(0, False)
        py0s.graph(1, True)

graphdispersion(-1/3, 1, 2, 20)

# findPhase(60,20, 20, "Files/phase_diagram.txt")

# findPhaseMag(0, 0.25, 20, 0, 12, 20, np.array([1, 1, 0]), 1, "phase_mag_110.txt")

# spinon_continuum(15, 15, -1/3)