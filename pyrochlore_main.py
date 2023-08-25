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
        py0s.findminLam()
        py0s.findLambda()
        # print(py0s.lams, py0s.minLams, py0s.condensed())
        # plt.axvline(x=py0s.lams[0], color='b', label='axvline - full height', linestyle='dashed')
        # plt.plot(py0s.lams[0], py0s.rho(py0s.lams)[0], marker='o')
        py0s.graph(False)
        # py0s.graphAlg(False)
        # py0s.graphAlg(False)
        # plt.legend(['Num', 'Alg'])
        plt.show()
    elif JP < 0:
        py0s = pypi.piFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
        py0s.findLambda()
        # temp = py0s.M_true(py0s.bigB)[:,0:4, 0:4] - np.conj(py0s.M_true(py0s.bigB)[:,4:8, 4:8])
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
        return 2*np.pi*np.array([1,0,0])
    elif mu == 1:
        return 2*np.pi*np.array([0,1,0])
    elif mu == 2:
        return 2*np.pi*np.array([0,0,1])

def BZbasisa(mu):
    if mu == 0:
        return 2*np.pi*np.array([-1,1,1])
    elif mu == 1:
        return 2*np.pi*np.array([1,-1,1])
    elif mu == 2:
        return 2*np.pi*np.array([1,1,-1])

def hkltoK(H, L):
    return np.einsum('ij,k->ijk',H, BZbasis(0)+BZbasis(1)) + np.einsum('ij,k->ijk',L, BZbasis(2))

def hkltoKtest(H, L):
    return np.einsum('ij,k->ijk',H, BZbasisa(0)+BZbasisa(1)) + np.einsum('ij,k->ijk',L, BZbasisa(2))

def SSSF_zero_cal(nK,h, n, BZres, Jpm, filename):
    py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, graphres=nK, h =h, n=n)
    py0s.findLambda()

    H = np.linspace(-2.5,2.5,nK)
    L = np.linspace(-2.5,2.5,nK)

    A, B = np.meshgrid(H, L)
    K = hkltoK(A,B).reshape((nK*nK,3))


    d1 = graph_SSSF_zero(py0s, K).reshape((nK, nK))
    np.savetxt("Files/"+filename+".txt", d1)

    GammaH = np.array([0, 0])
    LH =  np.array([1, 1])
    XH = np.array([0, 1])*2
    KH = np.array([3/4,0])*2
    UH = np.array([1 / 4, 1])*2
    UpH = np.array([1 / 4, -1])*2


    plt.text(U[0]+0.03,U[1]+0.03, '$U$')
    # d1 = np.loadtxt("Files/"+filename+".txt")
    plt.contourf(A,B, d1, levels=100)
    plt.ylabel(r'$(0,0,L)$')
    plt.xlabel(r'$(H,H,0)$')

    # dex = edges(d1, e, 5e-2)
    # plt.plot(kk, dex[0], 'b', kk, dex[1], 'b')
    plt.plot([0],[0], marker='o', color='k')
    plt.plot(np.linspace(XH, UH, 2).T[0], np.linspace(XH, UH, 2).T[1], marker='o', color='k')
    plt.plot(np.linspace(UH, LH, 2).T[0], np.linspace(UH, LH, 2).T[1], marker='o', color='k')
    plt.plot(np.linspace(KH, LH, 2).T[0], np.linspace(KH, LH, 2).T[1], marker='o', color='k')
    plt.plot(np.linspace(KH, UpH, 2).T[0], np.linspace(KH, UpH, 2).T[1], color='k')
    plt.plot(np.linspace(-UH, UpH, 2).T[0], np.linspace(-UH, UpH, 2).T[1], color='k')
    plt.plot(np.linspace(-UH, -KH, 2).T[0], np.linspace(-UH, -KH, 2).T[1], color='k')
    plt.plot(np.linspace(-UpH, -KH, 2).T[0], np.linspace(-UpH, -KH, 2).T[1], color='k')
    plt.plot(np.linspace(-UpH, UH, 2).T[0], np.linspace(-UpH, UH, 2).T[1], color='k')
    plt.text(GammaH[0]+0.03,GammaH[1]+0.03, '$\Gamma$')
    plt.text(LH[0]+0.03,LH[1]+0.03, '$L$')
    plt.text(XH[0]+0.03,XH[1]+0.03, '$X$')
    plt.text(KH[0]+0.03,KH[1]+0.03, '$K$')
    plt.text(UH[0] + 0.03, UH[1] + 0.03, '$U$')
    plt.savefig("Files/"+filename+".png")


def SSSF_pi_cal(nK,h, n, BZres, Jpm, filename):

    py0s = pypi.piFluxSolver(Jpm, BZres=BZres, graphres=nK,h=h,n=n)
    py0s.findLambda()
    # py0s.calAllDispersion()
    H = np.linspace(-2.5,2.5,nK)
    L = np.linspace(-2.5,2.5,nK)
    A, B = np.meshgrid(H, L)
    K = hkltoK(A,B)


    d1 = graph_SSSF_pi(py0s, K)
    np.savetxt("Files/"+filename+".txt", d1)


    Gamma = np.array([0, 0])
    L =  np.array([1, 1])/2
    X = np.array([0, 1])
    K = np.array([3/4,0])
    U = np.array([1 / 4, 1])
    Up = np.array([1 / 4, -1])



    # d1 = np.loadtxt("Files/spin_cont_test.txt")
    plt.contourf(A,B, d1, levels=100)
    # plt.pcolormesh(A,B, d1)
    plt.ylabel(r'$(0,0,L)$')
    plt.xlabel(r'$(H,H,0)$')

    plt.plot([0],[0], marker='o', color='k')
    plt.plot(np.linspace(X, U, 2).T[0], np.linspace(X, U, 2).T[1], marker='o', color='k')
    plt.plot(np.linspace(U, L, 2).T[0], np.linspace(U, L, 2).T[1], marker='o', color='k')
    plt.plot(np.linspace(K, L, 2).T[0], np.linspace(K, L, 2).T[1], marker='o', color='k')
    plt.plot(np.linspace(K, Up, 2).T[0], np.linspace(K, Up, 2).T[1], color='k')
    plt.plot(np.linspace(-U, Up, 2).T[0], np.linspace(-U, Up, 2).T[1], color='k')
    plt.plot(np.linspace(-U, -K, 2).T[0], np.linspace(-U, -K, 2).T[1], color='k')
    plt.plot(np.linspace(-Up, -K, 2).T[0], np.linspace(-Up, -K, 2).T[1], color='k')
    plt.plot(np.linspace(-Up, U, 2).T[0], np.linspace(-Up, U, 2).T[1], color='k')
    plt.text(Gamma[0]+0.03,Gamma[1]+0.03, '$\Gamma$')
    plt.text(L[0]+0.03,L[1]+0.03, '$L$')
    plt.text(X[0]+0.03,X[1]+0.03, '$X$')
    plt.text(K[0]+0.03,K[1]+0.03, '$K$')
    plt.text(U[0]+0.03,U[1]+0.03, '$U$')
    plt.savefig("Files/"+filename+".png")

def SSSF(nK,h, n, Jpm, BZres,  filename):
    if Jpm >= 0:
        SSSF_zero_cal(nK,h, n,BZres, Jpm, filename)
    else:
        SSSF_pi_cal(nK,h, n, BZres, Jpm, filename)

#endregion

h111=np.array([1,1,1])/np.sqrt(3)
h001=np.array([0,0,1])
h110 = np.array([1,1,0])/2

# graphdispersion(-1/3, 0, h111, 1, 2, 20, 20)
# graphdispersion(0.02,0.8, h111, 1, 2, 20, 20)
# graphdispersion(0.046, 0, h111, 1, 2, 20, 20)
# graphdispersion(0.05, 0, h111, 1, 2, 20, 20)
# graphdispersion(0.1, 0, h111, 1, 2, 20, 20)
# plt.show()

# findPhase(60,20, 20, "Files/phase_diagram.txt")

# PhaseMagtestH(0.0001, 0.25, 25, 0, 3, 25, h110, 35, 1, "0.txt")

# PhaseMagtestJP(0, 0.25, 25, 0, 3, 25, h111, 35, 1, "0.txt")
#
#


# spinon_continuum(50,50,50,0.046, 0.02,   "spin_con_zero_flux_final")
#
# # SSSF(25, 0, np.array([1,1,1]),0.02,25, "SSSF_zero_0.02_h111=0")
#
# # SSSF(25, 0, h111,0.06,25, "SSSF_zero_0.06")
# SSSF(25, 0, np.array([1,1,1]),-0.05,20, "SSSF_pi_-0.05")
SSSF(25, 0, np.array([1,1,1]),-0.40,40, "SSSF_pi_-0.40_1DETAILED")
# SSSF(25, 0, np.array([1,1,1]),-0.40,20, "SSSF_pi_-0.40")

# SSSF(25, 0.2, h111,0.02,25, "SSSF_zero_0.02_h111=0.2")
# SSSF(25, 0.4, h111,0.02,25, "SSSF_zero_0.02_h111=0.4")
# SSSF(25, 0.6, h111,0.02,25, "SSSF_zero_0.02_h111=0.6")
#
# SSSF(25, 0.2, h001,0.02,25, "SSSF_zero_0.02_h001=0.2")
# SSSF(25, 0.4, h001,0.02,25, "SSSF_zero_0.02_h001=0.4")
# SSSF(25, 0.6, h001,0.02,25, "SSSF_zero_0.02_h001=0.6")
# #
# SSSF(25, 0.8, h110,0.02,25, "SSSF_zero_0.02_h110=0.8")
# SSSF(25, 1.6, h110,0.02,25, "SSSF_zero_0.02_h110=1.6")
# SSSF(25, 2.4, h110,0.02,25, "SSSF_zero_0.02_h110=2.4")
# #
#
# SSSF(25, 0, np.array([1,1,1]),-0.25,25, "SSSF_pi_-0.25_dumb")
# SSSF(25, 0, np.array([1,1,1]),-0.05,25, "SSSF_pi_-0.05_dumb")

#
# graphPhase("Files/phase_diagram.txt")


# findPhaseMag(0, 0.25, 35, 0, 3, 35, h111, 35, 1, "phase_test_111_kappa=1")
# findPhaseMag(0, 0.25, 35, 0, 3, 35, h001, 35, 1, "phase_test_001_kappa=1")
# findPhaseMag(0, 0.25, 35, 0, 12, 35, h110, 35, 1, "phase_test_110_kappa=1")
# graphMagPhase("phase_test_111_kappa=1", 0.25,3)
# graphMagPhase("phase_test_001_kappa=1", 0.25,3)
# graphMagPhase("phase_test_110_kappa=1", 0.25,12)

# graphMagPhase("phase_test_001", 0.25,3)
# graphMagPhase("phase_test_110", 0.25,12)