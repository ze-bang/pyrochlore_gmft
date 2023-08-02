import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
import math
import time
import sys
from numba import jit

#region miscellaneous
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
#endregion

#region graph dispersion
def graphdispersion(JP, kappa, rho, graphres, BZres):
    if JP >= 0:
        py0s = py0.zeroFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres)
        py0s.setupALL()
        py0s.findLambda()
        print(py0s.lams)
        py0s.graph(0, False)
        py0s.graph(1, True)
    elif JP < 0:
        py0s = pypi.piFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres)
        py0s.setupALL()
        py0s.findLambda()
        print(py0s.M_pi_single(np.pi * np.array([1, 1, 1])/2, 0))
        print(py0s.M_pi_single(np.pi*np.array([1,1,-1])/2, 0))
        print(py0s.M_pi_single(np.pi * np.array([-1, -1, 1])/2, 0))
        print(py0s.M_pi_single(np.pi*np.array([-1,-1,-1])/2, 0))
        py0s.graph(0, False)
        py0s.graph(1, True)
#endregion

#region Phase for Anisotropy
def findPhase(nK, nE, res, filename):

    JP = np.linspace(0, 0.03, nK)
    kappaR = np.linspace(-1, 0, nE)
    kappa = (1+kappaR)/(1-kappaR)

    #
    # JP = [-0.3]
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
                pyps = pypi.piFluxSolver(JP[i], kappa=kappa[j], BZres=res, lam=abs(JP[i])*2)
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
            end = time.time()
            el = (end - start)*(totaltask-count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            sys.stdout.flush()



    np.savetxt(filename, phases)
#endregion

#region Phase for Magnetic Field
def findPhaseMag(JPm, JPmax, nK, hm, hmax, nH, n, BZres, kappa, filename):
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
                py0s = py0.zeroFluxSolver(JP[i], h = h[j], n=n, kappa=kappa, BZres=BZres)
                py0s.setupALL()
                print("Finding 0 Flux Lambda")
                # phases[i][j] = py0s.phase_test()
                py0s.findLambda()
                print(py0s.lams, py0s.minLams)
                phases[i,j] = phaseMag(py0s.lams, py0s.minLams, 0)
                # try:
                #     py0s.calDispersion()
                #     phases[i][j] = 1
                # except:
                #     phases[i][j] = 0
            else:
                pyps = pypi.piFluxSolver(JP[i], h= h[j], n=n, kappa=kappa)
                pyps.setupALL()
                print("Finding pi Flux Lambda")
                pyps.findLambda()
                print([pyps.lams, pyps.minLams])
                try:
                    pyps.calDispersion()
                    phases[i][j] = 1
                except:
                    phases[i][j] = 0

    np.savetxt(filename, phases)
#endregion

#region DSSF
def spinon_continuum_zero(nE, nK, Jpm, filename, BZres):
    py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, graphres=nK)
    py0s.setupALL()
    py0s.findLambda()
    py0s.calDispersion()

    e = np.linspace(py0s.gap(0), py0s.EMAX(0)*2.1, nE)
    kk = np.concatenate((np.linspace(-0.5, 0, nK), np.linspace(0, 0.3, nK), np.linspace(0.3, 0.5, nK), np.linspace(0.5,0.9, nK), np.linspace(0.9, 1.3, nK), np.linspace(1.3, 1.6, nK), np.linspace(1.6, 1.85, nK)))
    d1 = graph_spin_cont_zero(py0s, e, np.concatenate((py0s.GammaX, py0s.XW, py0s.WK, py0s.KGamma, py0s.GammaL, py0s.LU, py0s.UW)), 0.02)


    np.savetxt("Files/"+filename+".txt", d1)

    # d1 = np.loadtxt("Files/spin_cont_test.txt")

    X,Y = np.meshgrid(kk, e)
    plt.contourf(X,Y, d1, levels=100)
    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=-0.5, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.3, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.5, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.9, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.3, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.6, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.85, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [-0.5, 0, 0.3, 0.5, 0.9, 1.3, 1.6, 1.85]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)
    # dex = edges(d1, e, 5e-2)
    # plt.plot(kk, dex[0], 'b', kk, dex[1], 'b')
    plt.savefig("Files/"+filename+".png")
    # plt.show()

def spinon_continuum_pi(nE, nK, Jpm, filename, BZres):

    py0s = pypi.piFluxSolver(Jpm, BZres=BZres, graphres=nK)
    py0s.setupALL()
    py0s.findLambda()
    py0s.calAllDispersion()

    e = np.linspace(py0s.gap(0), py0s.EMAX(0)*2.1, nE)
    kk = np.concatenate((np.linspace(-0.5, 0, nK), np.linspace(0, 0.3, nK), np.linspace(0.3, 0.5, nK), np.linspace(0.5,0.9, nK), np.linspace(0.9, 1.3, nK), np.linspace(1.3, 1.6, nK), np.linspace(1.6, 1.85, nK)))
    d1 = graph_spin_cont_pi(py0s, e, np.concatenate((py0s.GammaX, py0s.XW, py0s.WK, py0s.KGamma, py0s.GammaL, py0s.LU, py0s.UW)), 1e-4)
    # d1 = graph_spin_cont_pi(py0s, e, py0s.GammaX, 0.02)
    # kk = np.linspace(-0.5, 0, nK)


    np.savetxt("Files/"+filename+".txt", d1)

    # d1 = np.loadtxt("Files/spin_cont_test.txt")

    X,Y = np.meshgrid(kk, e)
    plt.contourf(X,Y, d1, levels=100)
    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=-0.5, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.3, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.5, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.9, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.3, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.6, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.85, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [-0.5, 0, 0.3, 0.5, 0.9, 1.3, 1.6, 1.85]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)
    # dex = edges(d1, e, 5e-2)
    # plt.plot(kk, dex[0], 'b', kk, dex[1], 'b')
    plt.savefig("Files/"+filename+".png")
    # plt.show()

def spinon_continuum(nE, nK, BZres, Jpm, filename):
    if Jpm >= 0:
        spinon_continuum_zero(nE, nK, Jpm, filename, BZres)
    else:
        spinon_continuum_pi(nE, nK, Jpm, filename, BZres)

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

def SSSF_zero_cal(nK, BZres, Jpm, filename):
    py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, graphres=nK)
    py0s.setupALL()
    py0s.findLambda()
    py0s.calDispersion()

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


def SSSF_pi_cal(nK, BZres, Jpm, filename):

    py0s = pypi.piFluxSolver(Jpm, BZres=BZres, graphres=nK)
    py0s.setupALL()
    py0s.findLambda()
    # py0s.calAllDispersion()

    H = np.linspace(-2,2,nK)
    L = np.linspace(-2,2,nK)
    A, B = np.meshgrid(H, L)
    K = hkltoKtest(A,B).reshape((nK*nK,3))


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

def SSSF(nK, BZres, Jpm, filename):
    if Jpm >= 0:
        SSSF_zero_cal(nK,BZres, Jpm, filename)
    else:
        SSSF_pi_cal(nK, BZres, Jpm, filename)

#endregion

# graphdispersion(-1/3, 1, 2, 20, 20)

# findPhase(60,20, 20, "Files/phase_diagram.txt")

# findPhaseMag(0.0, 0.25, 10, 0, 3, 30, np.array([1,1,1]), 20, 1, "phase_mag_111.txt")

spinon_continuum(50,50,50,0.046, "spin_con_zero_test")

# SSSF(20,20,-1/3, "SSSF_pi_test")