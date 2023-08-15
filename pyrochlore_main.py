import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
import math
import time
import sys
from graph import *
from numba import jit
from misc_helper import *
#region miscellaneous


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
    elif lamA - minLams[0] < 0:
        print("PMu Phase")
        return 1+pi
    elif lamB - minLams[1]< 0:
        print("PMu Phase")
        return -(1+pi)
    else:
        print("QSL Phase")
        print(3*(-1)**pi)
        return 3*(-1)**pi

#endregion

#region graph dispersion
def graphdispersion(JP,h, n, kappa, rho, graphres, BZres):
    if JP >= 0:
        py0s = py0.zeroFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
        py0s.findLambda()
        py0s.graph(False)
        # py0s.graphAlg(False)
        # plt.legend(['Num', 'Alg'])
        plt.show()
    elif JP < 0:
        py0s = pypi.piFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
        py0s.findLambda()
        py0s.graph(True)
#endregion

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
                pyps.setupALL()
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



    np.savetxt(filename, phases)
#endregion

#region Phase for Magnetic Field

def PhaseMagtestJP(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):

    JP = np.linspace(JPm, JPmax, nK)

    gap = np.zeros(nK)
    dev = np.zeros(nK)
    E111 = np.zeros(nK)
    E000 = np.zeros(nK)
    # for i in range (nH):
    for i in range (nK):
        print("Jpm is now " + str(JP[i]))
        # print("h is now " + str(h[j]))
        if JP[i] >= 0:
            py0s = py0.zeroFluxSolver(JP[i], h = 0, n=n, kappa=kappa, BZres=BZres)
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
            dev[i] = py0s.rho_dev()
            E111[i] = np.min(py0s.E_single(np.pi*np.array([1,1,1])))
            E000[i] = np.min(py0s.E_single(np.array([0,0,0])))
            print([gap[i], E111[i], E000[i]])
    plt.plot(JP, gap, color='b')
    plt.plot(JP, E111, color='g')
    plt.plot(JP, E000, color='r')
    plt.plot(JP, dev, color='black')
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

    JP = np.linspace(JPm, JPmax, nK)
    h = np.linspace(hm, hmax, nH)
    phases = np.zeros((nK, nH), dtype=float)
    dev = np.zeros((nK, nH), dtype=float)

    for i in range (nK):
        for j in range (nH):
            print("Jpm is now " + str(JP[i]))
            print("h is now " + str(h[j]))
            if JP[0] >= 0:
                py0s = py0.zeroFluxSolver(JP[i], h = h[j], n=n, kappa=kappa, BZres=BZres)
                print("Finding 0 Flux Lambda")
                # phases[i][j] = py0s.phase_test()
                py0s.findLambda()
                phases[i, j] = py0s.gap()
                dev[i, j] = py0s.rho_dev()
    np.savetxt(filename+'.txt', phases)
    np.savetxt(filename+'_dev.txt', dev)
#endregion

#region DSSF
def spinon_continuum_zero(nE, nK, Jpm, filename, BZres, tol):
    py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, graphres=nK)
    py0s.findLambda()
    e = np.linspace(py0s.gap(), py0s.EMAX()*2.1, nE)
    kk = np.concatenate((np.linspace(gGamma1, gX, len(GammaX)), np.linspace(gX, gW1, len(XW)), np.linspace(gW1, gK, len(WK))
                         , np.linspace(gK,gGamma2, len(KGamma)), np.linspace(gGamma2, gL, len(GammaL)), np.linspace(gL, gU, len(LU)), np.linspace(gU, gW2, len(UW))))
    d1 = graph_spin_cont_zero(py0s, e, np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW)), tol)


    np.savetxt("Files/"+filename+".txt", d1)

    # d1 = np.loadtxt("Files/"+filename+".txt")

    X,Y = np.meshgrid(kk, e)
    plt.contourf(X,Y, d1, levels=100)
    plt.ylabel(r'$\omega/J_{zz}$')
    py0s.graph_loweredge(False)
    py0s.graph_upperedge(False)
    plt.savefig("Files/"+filename+".png")
    plt.show()

def spinon_continuum_pi(nE, nK, Jpm, filename, BZres, tol):

    py0s = pypi.piFluxSolver(Jpm, BZres=BZres, graphres=nK)
    py0s.findLambda()

    e = np.linspace(py0s.gap(), py0s.EMAX()*2.1, nE)
    kk = np.concatenate((np.linspace(gGamma1, gX, nK), np.linspace(gX, gW1, nK), np.linspace(gW1, gK, nK), np.linspace(gK,gGamma2, nK), np.linspace(gGamma2, gL, nK), np.linspace(gL, gU, nK), np.linspace(gU, gW2, nK)))
    d1 = graph_spin_cont_pi(py0s, e, np.concatenate((py0s.GammaX, py0s.XW, py0s.WK, py0s.KGamma, py0s.GammaL, py0s.LU, py0s.UW)), tol)
    # d1 = graph_spin_cont_pi(py0s, e, py0s.GammaX, 0.02)
    # kk = np.linspace(gGamma1, gX, nK)


    np.savetxt("Files/"+filename+".txt", d1)

    # d1 = np.loadtxt("Files/spin_cont_test.txt")

    X,Y = np.meshgrid(kk, e)
    plt.contourf(X,Y, d1, levels=100)
    plt.ylabel(r'$\omega/J_{zz}$')
    # py0s.graph_loweredge(False)
    # plt.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=gX, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=gW1, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=gK, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=gGamma2, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=gL, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=gU, color='b', label='axvline - full height', linestyle='dashed')
    # plt.axvline(x=gW2, color='b', label='axvline - full height', linestyle='dashed')
    # xlabpos = [gGamma1, gX, gW1, gK, gGamma2, gL, gU, gW2]
    # labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    # plt.xticks(xlabpos, labels)
    # dex = edges(d1, e, 5e-2)
    # plt.plot(kk, dex[0], 'b', kk, dex[1], 'b')
    plt.savefig("Files/"+filename+".png")
    plt.show()

def spinon_continuum(nE, nK, BZres, Jpm, tol, filename):
    if Jpm >= 0:
        spinon_continuum_zero(nE, nK, Jpm, filename, BZres, tol)
    else:
        spinon_continuum_pi(nE, nK, Jpm, filename, BZres, tol)

#endregion

#region SSSF

def BZbasis(mu):
    if mu == 0:
        return np.pi*np.array([1,0,0])
    elif mu == 1:
        return np.pi*np.array([0,1,0])
    elif mu == 2:
        return np.pi*np.array([0,0,1])

def BZbasisa(mu):
    if mu == 0:
        return np.pi*np.array([1,0,0])/2
    elif mu == 1:
        return np.pi*np.array([0,1,0])/2
    elif mu == 2:
        return np.pi*np.array([0,0,1])/2

def hkltoK(H, L):
    return np.einsum('ij,k->ijk',H, BZbasis(0)+BZbasis(1)) + np.einsum('ij,k->ijk',L, BZbasis(2))

def hkltoKtest(H, L):
    return np.einsum('ij,k->ijk',H, BZbasisa(0)+BZbasisa(1)) + np.einsum('ij,k->ijk',L, BZbasisa(2))

def SSSF_zero_cal(nK,h, n, BZres, Jpm, filename):
    py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, graphres=nK, h =h, n=n)
    py0s.findLambda()

    H = np.linspace(-2,2,nK)
    L = np.linspace(-2,2,nK)

    A, B = np.meshgrid(H, L)
    K = hkltoK(A,B).reshape((nK*nK,3))


    d1 = graph_SSSF_zero(py0s, K).reshape((nK, nK))
    np.savetxt("Files/"+filename+".txt", d1)

    Gamma = np.array([0, 0])
    L =  np.array([1, 1])/2
    X = np.array([0, 1])
    K = np.array([3/4,0])
    U = np.array([1 / 4, 1])
    Up = np.array([1 / 4, -1])

    plt.plot([0],[0], marker='o', color='b')
    plt.plot(drawLine(X, U, 2).T[0], drawLine(X, U, 2).T[1], marker='o', color='b')
    plt.plot(drawLine(U, L, 2).T[0], drawLine(U, L, 2).T[1], marker='o', color='b')
    plt.plot(drawLine(K, L, 2).T[0], drawLine(K, L, 2).T[1], marker='o', color='b')
    plt.plot(drawLine(K, Up, 2).T[0], drawLine(K, Up, 2).T[1], color='b')
    plt.plot(drawLine(-U, Up, 2).T[0], drawLine(-U, Up, 2).T[1], color='b')
    plt.plot(drawLine(-U, -K, 2).T[0], drawLine(-U, -K, 2).T[1], color='b')
    plt.plot(drawLine(-Up, -K, 2).T[0], drawLine(-Up, -K, 2).T[1], color='b')
    plt.plot(drawLine(-Up, U, 2).T[0], drawLine(-Up, U, 2).T[1], color='b')
    plt.text(Gamma[0]+0.03,Gamma[1]+0.03, '$\Gamma$')
    plt.text(L[0]+0.03,L[1]+0.03, '$L$')
    plt.text(X[0]+0.03,X[1]+0.03, '$X$')
    plt.text(K[0]+0.03,K[1]+0.03, '$K$')
    plt.text(U[0]+0.03,U[1]+0.03, '$U$')
    # d1 = np.loadtxt("Files/spin_cont_test.txt")
    plt.contourf(A,B, d1, levels=100)
    plt.ylabel(r'$(0,0,L)$')
    plt.xlabel(r'$(H,H,0)$')

    # dex = edges(d1, e, 5e-2)
    # plt.plot(kk, dex[0], 'b', kk, dex[1], 'b')
    plt.savefig("Files/"+filename+".png")
    plt.show()


def SSSF_pi_cal(nK,h, n, BZres, Jpm, filename):

    py0s = pypi.piFluxSolver(Jpm, BZres=BZres, graphres=nK,h=h,n=n)
    py0s.findLambda()
    # py0s.calAllDispersion()

    H = np.linspace(-2,2,nK)
    L = np.linspace(-2,2,nK)
    A, B = np.meshgrid(H, L)
    K = hkltoK(A,B).reshape((nK*nK,3))


    d1 = graph_SSSF_pi(py0s, K).reshape((nK, nK))
    np.savetxt("Files/"+filename+".txt", d1)


    Gamma = np.array([0, 0])
    L =  np.array([1, 1])/2
    X = np.array([0, 1])
    K = np.array([3/4,0])
    U = np.array([1 / 4, 1])
    Up = np.array([1 / 4, -1])

    plt.plot([0],[0], marker='o', color='b')
    plt.plot(drawLine(X, U, 2).T[0], drawLine(X, U, 2).T[1], marker='o', color='b')
    plt.plot(drawLine(U, L, 2).T[0], drawLine(U, L, 2).T[1], marker='o', color='b')
    plt.plot(drawLine(K, L, 2).T[0], drawLine(K, L, 2).T[1], marker='o', color='b')
    plt.plot(drawLine(K, Up, 2).T[0], drawLine(K, Up, 2).T[1], color='b')
    plt.plot(drawLine(-U, Up, 2).T[0], drawLine(-U, Up, 2).T[1], color='b')
    plt.plot(drawLine(-U, -K, 2).T[0], drawLine(-U, -K, 2).T[1], color='b')
    plt.plot(drawLine(-Up, -K, 2).T[0], drawLine(-Up, -K, 2).T[1], color='b')
    plt.plot(drawLine(-Up, U, 2).T[0], drawLine(-Up, U, 2).T[1], color='b')
    plt.text(Gamma[0]+0.03,Gamma[1]+0.03, '$\Gamma$')
    plt.text(L[0]+0.03,L[1]+0.03, '$L$')
    plt.text(X[0]+0.03,X[1]+0.03, '$X$')
    plt.text(K[0]+0.03,K[1]+0.03, '$K$')
    plt.text(U[0]+0.03,U[1]+0.03, '$U$')

    # d1 = np.loadtxt("Files/spin_cont_test.txt")
    plt.contourf(A,B, d1, levels=100)
    plt.ylabel(r'$(0,0,L)$')
    plt.xlabel(r'$(H,H,0)$')
    # dex = edges(d1, e, 5e-2)
    # plt.plot(kk, dex[0], 'b', kk, dex[1], 'b')
    plt.savefig("Files/"+filename+".png")
    plt.show()

def SSSF(nK,h, n, Jpm, BZres,  filename):
    if Jpm >= 0:
        SSSF_zero_cal(nK,h, n,BZres, Jpm, filename)
    else:
        SSSF_pi_cal(nK,h, n, BZres, Jpm, filename)

#endregion

h111=np.array([1,1,1])/np.sqrt(3)
h001=np.array([0,0,1])
h110 = np.array([1,1,0])/2


# graphdispersion(0.046, 0, h111, 1, 2, 20, 20)

# findPhase(60,20, 20, "Files/phase_diagram.txt")

PhaseMagtestH(0.0001, 0.25, 25, 0, 3, 25, h110, 35, 1, "0.txt")

# PhaseMagtestJP(0, 0.25, 50, 0, 3, 25, h111, 35, 1, "0.txt")


# findPhaseMag(0, 0.25, 25, 0, 3, 25, h111, 25, 1, "phase_test_111")
# findPhaseMag(0, 0.25, 25, 0, 3, 25, h001, 35, 1, "phase_test_001")
# findPhaseMag(0, 0.25, 25, 0, 6, 25, h110, 35, 1, "phase_test_110")

# spinon_continuum(20,20,20,0.046, 0.04, "spin_con_zero_flux_final")

# SSSF(20, 0, np.array([1,1,1]),-0.2,20, "SSSF_zero_test")

# graphPhase("Files/phase_diagram.txt")
# graphMagPhase("phase_test_111", 0.25,3)
# graphMagPhase("phase_test_111_dev", 0.25,3)
# graphMagPhase("phase_test_001", 0.25,3)
# graphMagPhase("phase_test_110", 0.25,6)