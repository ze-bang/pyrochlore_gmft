import numpy as np
import matplotlib.pyplot as plt
import warnings
from numba import jit
from numba.experimental import jitclass
from misc_helper import *


def exponent_pi(k, alpha, mu, nu, rs1, rs2):
    rs = rs1 - neta(alpha) * step(mu)

    f = np.real(np.exp(1j * neta(alpha) * (A_pi(rs, rs2) - A_pi(rs, rs1))))
    return f * np.exp(1j * neta(alpha) * (np.dot(k, NN(nu) - NN(mu))))


def M_pi_term(k, alpha, rs1, rs2, mu, nu, eta, Jpm):
    temp = -Jpm / 4 * eta[alpha] * exponent_pi(k, alpha, mu, nu, rs1, rs2)
    return temp


def M_pi_mag_term(k, alpha, rs1, mu, h, n):
    rs = rs1 - neta(alpha) * step(mu)
    temp = -1 / 4 * h * np.dot(n, z(mu)) * np.exp(1j * A_pi(rs, rs1)) * np.exp(
        1j * np.dot(k, neta(alpha) * NN(mu)))
    return temp

def M_pi_mag_sub(k, rs, alpha, h, n):
    M = np.zeros((len(k), 4, 4), dtype=complex)
    for i in range(4):
        nu = unitCell(rs) + neta(alpha) * step(i)
        rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
        index2 = findS(rs2)
        M[:,rs,index2] += M_pi_mag_term(k, alpha, rs2, i, h, n)
    return M
def M_pi_sub(k, rs, alpha, eta, Jpm):
    M = np.zeros((len(k), 4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            if not i == j:
                mu = unitCell(rs) + neta(alpha) * step(i)
                nu = unitCell(rs) + neta(alpha) * step(j)
                rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                index1 = findS(rs1)
                index2 = findS(rs2)
                M[:, index1, index2] += M_pi_term(k, alpha, mu, nu, i, j, eta, Jpm)

    return M


def M_pi(k,eta,Jpm, h, n):
    bigM1 = np.zeros((len(k), 4, 4, 4), dtype=complex)
    bigM2 = np.zeros((len(k), 4, 4, 4), dtype=complex)
    bigMag1 = np.zeros((len(k), 4, 4, 4), dtype=complex)
    for i in range(4):
        bigM1[:, i, :, :] = M_pi_sub(k, i, 0,eta,Jpm)
        bigM2[:, i, :, :] = M_pi_sub(k, i, 1,eta,Jpm)
        bigMag1[:, i, :, :] = M_pi_mag_sub(k, i, 0,h,n)
    M1 = np.einsum('ijkl->ikl', bigM1)
    M2 = np.einsum('ijkl->ikl', bigM2)
    Mag1 = np.einsum('ijkl->ikl', bigMag1)
    Mag2 = np.transpose(np.conj(Mag1), (0,2,1))
    FM = np.block([[M1, Mag1], [Mag2, M2]])
    return FM


def M_pi_sub_single(k, rs, alpha, eta, Jpm, h, n):
    M = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            if not i == j:
                mu = unitCell(rs) + neta(alpha) * step(i)
                nu = unitCell(rs) + neta(alpha) * step(j)
                rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                index1 = findS(rs1)
                index2 = findS(rs2)
                M[index1, index2] += M_pi_term(k, alpha, mu, nu, i, j, eta, Jpm)
                M[:,i,index2] += -M_pi_mag_term(k, alpha, rs2, j, h, n)
                M[:, index2, i] += -np.conj(M_pi_mag_term(k, 1-alpha, rs2, j, h, n))
    return M


def M_pi_single(k, eta, Jpm, h, n):
    bigM = np.zeros((4, 4, 4), dtype=complex)
    bigM2 = np.zeros((4, 4, 4), dtype=complex)
    for i in range(4):
        bigM[i, :, :] = M_pi_sub_single(k, i, 0, eta, Jpm, h, n)
        bigM2[i, :, :] = M_pi_sub_single(k, i, 1, eta, Jpm, h, n)
    M = np.einsum('ijk->jk', bigM)
    M1 = np.einsum('ijk->jk', bigM2)
    FM = np.block([[M, np.zeros((4, 4))], [np.zeros((4, 4)), M1]])
    return FM

def E_pi_fixed(lams, M):
    M = M + np.diag(np.repeat(lams,4))
    E, V = np.linalg.eigh(M)
    return [E, V]


def E_pi(lams, k, eta, Jpm, h, n):
    M = M_pi(k,eta,Jpm, h, n)
    M = M + np.diag(np.repeat(lams,4))
    E, V = np.linalg.eigh(M)
    return [E,V]


def rho_true(Jzz, M, lams):
    dumb = np.array([[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1]])
    temp = M + np.diag(np.repeat(lams,4))
    E, V = np.linalg.eigh(temp)
    Vt = np.real(np.einsum('ijk,ijk->ijk',V, np.conj(V)))
    Ep = np.einsum('ijk, ik->ij', Vt, 1/np.sqrt(2*Jzz*E))
    Ep = np.mean(np.einsum('jk, ik->ij', dumb, Ep), axis=0)/4
    return Ep




def findlambda_pi(M, Jzz, kappa, tol):
    warnings.filterwarnings("error")
    lamMin = np.zeros(2)
    lamMax = 10*np.ones(2)
    lams = (lamMin + lamMax) / 2
    rhoguess = rho_true(Jzz, M, lams)
    # print(self.kappa)
    yes = True

    while yes >= tol:
        # for i in range(2):
         lams= (lamMin+lamMax)/2
         # rhoguess = rho_true(Jzz, M, lams)
         try:
             rhoguess = rho_true(Jzz, M, lams)
             # rhoguess = self.rho_zero(alpha, self.lams)
             for i in range(2):
                 if rhoguess[i] - kappa > 0:
                     lamMin[i]  = lams[i]
                 else:
                     lamMax[i]  = lams[i]
         except:
             lamMin = lams
             # if lamMax == 0:
             #     break
         if np.absolute(rhoguess[0]-kappa)<=tol and np.absolute(rhoguess[1]-kappa)<=tol:
            yes = False
    return lams



# graphing BZ

def dispersion_pi(lams, k, Jzz, Jpm, eta, h, n):
    temp = np.sqrt(2*Jzz*E_pi(lams, k, eta, Jpm, h, n)[0])
    return temp

def calDispersion(lams, Jzz, Jpm, eta, h, n):
    dGammaX= dispersion_pi(lams, GammaX, Jzz, Jpm, eta, h, n)
    dXW= dispersion_pi(lams, XW, Jzz, Jpm, eta, h, n)
    dWK = dispersion_pi(lams, WK, Jzz, Jpm, eta, h, n)
    dKGamma = dispersion_pi(lams, KGamma, Jzz, Jpm, eta, h, n)
    dGammaL = dispersion_pi(lams, GammaL, Jzz, Jpm, eta, h, n)
    dLU= dispersion_pi(lams, LU, Jzz, Jpm, eta, h, n)
    dUW = dispersion_pi(lams, UW, Jzz, Jpm, eta, h, n)

    for i in range(8):
        plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX[:,i], 'b')
        plt.plot(np.linspace(gX, gW1, len(dXW)), dXW[:, i] , 'b')
        plt.plot(np.linspace(gW1, gK, len(dWK)), dWK[:, i], 'b')
        plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma[:, i], 'b')
        plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL[:, i], 'b')
        plt.plot(np.linspace(gL, gU, len(dLU)), dLU[:, i], 'b')
        plt.plot(np.linspace(gU, gW2, len(dUW)),dUW[:, i], 'b')

    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW1, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gK, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma2, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gL, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gU, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW2, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [gGamma1,gX,gW1,gK,gGamma2,gL,gU,gW2]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)

#
# def minCal(lams, q, Jzz, Jpm, eta, h, n, K):
#     temp = np.zeros((len(q),2))
#     mins = np.sqrt(2 * Jzz * E_zero_true(lams, K, Jpm, eta, h, n)[0])
#     for i in range(len(q)):
#         temp[i] = np.amin(np.sqrt(2 * Jzz * E_zero_true(lams, q[i]+K, Jpm, eta, h, n)[0]) + mins, axis=0)
#     return temp
#
# def maxCal(lams, q, Jzz, Jpm, eta, h, n, K):
#     temp = np.zeros((len(q),2))
#     mins = np.sqrt(2 * Jzz * E_zero_true(lams, K, Jpm, eta, h, n)[0])
#     for i in range(len(q)):
#         temp[i] = np.amax(np.sqrt(2 * Jzz * E_zero_true(lams, q[i]+K, Jpm, eta, h, n)[0]) + mins, axis=0)
#     return temp
#
# def loweredge(lams, Jzz, Jpm, eta, h, n, K):
#     dGammaX= minCal(lams, GammaX, Jzz, Jpm, eta, h, n, K)
#     dXW= minCal(lams, XW, Jzz, Jpm, eta, h, n, K)
#     dWK = minCal(lams, WK, Jzz, Jpm, eta, h, n, K)
#     dKGamma = minCal(lams, KGamma, Jzz, Jpm, eta, h, n, K)
#     dGammaL = minCal(lams, GammaL, Jzz, Jpm, eta, h, n, K)
#     dLU= minCal(lams, LU, Jzz, Jpm, eta, h, n, K)
#     dUW = minCal(lams, UW, Jzz, Jpm, eta, h, n, K)
#
#     for i in range(8):
#         plt.plot(np.linspace(gGamma1, gX, graphres), dGammaX[:,i], 'w')
#         plt.plot(np.linspace(gX, gW1, graphres), dXW[:, i] , 'w')
#         plt.plot(np.linspace(gW1, gK, graphres), dWK[:, i], 'w')
#         plt.plot(np.linspace(gK, gGamma2, graphres), dKGamma[:, i], 'w')
#         plt.plot(np.linspace(gGamma2, gL, graphres), dGammaL[:, i], 'w')
#         plt.plot(np.linspace(gL, gU, graphres), dLU[:, i], 'w')
#         plt.plot(np.linspace(gU, gW2, graphres),dUW[:, i], 'w')
#     plt.ylabel(r'$\omega/J_{zz}$')
#     plt.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gW, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gW, color='w', label='axvline - full height', linestyle='dashed')
#     xlabpos = [gGamma1,gX,gW1,gK,gGamma2,gL,gU,gW2]
#     labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
#     plt.xticks(xlabpos, labels)
#
# def upperedge(lams, Jzz, Jpm, eta, h, n, K):
#     dGammaX= maxCal(lams, GammaX, Jzz, Jpm, eta, h, n, K)
#     dXW= maxCal(lams, XW, Jzz, Jpm, eta, h, n, K)
#     dWK = maxCal(lams, WK, Jzz, Jpm, eta, h, n, K)
#     dKGamma = maxCal(lams, KGamma, Jzz, Jpm, eta, h, n, K)
#     dGammaL = maxCal(lams, GammaL, Jzz, Jpm, eta, h, n, K)
#     dLU= maxCal(lams, LU, Jzz, Jpm, eta, h, n, K)
#     dUW = maxCal(lams, UW, Jzz, Jpm, eta, h, n, K)
#
#     for i in range(8):
#         plt.plot(np.linspace(gGamma1, gX, graphres), dGammaX[:,i], 'w')
#         plt.plot(np.linspace(gX, gW1, graphres), dXW[:, i] , 'w')
#         plt.plot(np.linspace(gW1, gK, graphres), dWK[:, i], 'w')
#         plt.plot(np.linspace(gK, gGamma2, graphres), dKGamma[:, i], 'w')
#         plt.plot(np.linspace(gGamma2, gL, graphres), dGammaL[:, i], 'w')
#         plt.plot(np.linspace(gL, gU, graphres), dLU[:, i], 'w')
#         plt.plot(np.linspace(gU, gW2, graphres),dUW[:, i], 'w')
#     plt.ylabel(r'$\omega/J_{zz}$')
#     plt.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gW, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=gW, color='w', label='axvline - full height', linestyle='dashed')
#     xlabpos = [gGamma1,gX,gW1,gK,gGamma2,gL,gU,gW2]
#     labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
#     plt.xticks(xlabpos, labels)

def gap(M, lams):
    temp = M + np.diag(np.repeat(lams,4))
    E,V = np.linalg.eigh(temp)
    # E = np.sqrt(E)
    temp = np.amin(E)
    print("Gap is " + str(temp))
    return temp

def EMAX(M, lams):
    temp = M + np.diag(np.repeat(lams,4))
    E,V = np.linalg.eigh(temp)
    temp = np.amax(E)
    return temp

#
# def GS(lams, k, Jzz, Jpm, eta, h, n):
#     return np.mean(dispersion_pi(lams, k, Jzz, Jpm, eta), axis=0) - np.repeat(lams)

class piFluxSolver:

    def __init__(self, Jpm, h=0, n=np.array([0,0,0]), eta=1, kappa=2, lam=2, BZres=20, graphres=20, Jzz=1):
        self.Jzz = Jzz
        self.Jpm = Jpm
        self.kappa = kappa
        self.eta = np.array([eta, 1], dtype=float)
        self.tol = 1e-3
        self.lams = np.array([lam, lam], dtype=float)
        self.h = h
        self.n = n


        self.minLams = np.zeros(2)

        self.BZres = BZres
        self.graphres = graphres
        self.bigB = np.concatenate((genBZ(BZres), symK))
        self.MF = M_pi(self.bigB, self.eta, self.Jpm, self.h, self.n)

    #alpha = 1 for A = -1 for B


    def findLambda(self):
        self.lams = findlambda_pi(self.MF, self.Jzz, self.kappa, self.tol)
        warnings.resetwarnings()

    def M_true(self, k):
        return M_pi(k, self.eta, self.Jpm, self.h, self.n)

    def M_pi_sub(self, k, rs, alpha):
        return M_pi_sub(k, rs, alpha, self.eta, self.Jpm)

    def E_pi(self, k):
        return np.sqrt(2*self.Jzz*E_pi(self.lams, k, self.eta, self.Jpm, self.h, self.n)[0])

    def dispersion(self, k):
        return dispersion_pi(self.lams, k, self.Jzz, self.Jpm, self.eta, self.h, self.n)
    def LV_zero(self, k):
        return E_pi(self.lams, k, self.eta, self.Jpm, self.h, self.n)

    def gap(self):
        return np.sqrt(2*self.Jzz*gap(self.MF, self.lams))

    def graph(self, show):
        print(self.eta)
        calDispersion(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n)
        if show:
            plt.show()

    def E_single(self, k):
        M = M_pi_single(k, self.eta, self.Jpm, self.h, self.n) + np.diag(np.repeat(self.lams, 4))
        E, V = np.linalg.eigh(M)
        return np.sqrt(2*self.Jzz*E)

    def EMAX(self):
        return np.sqrt(2*self.Jzz*EMAX(self.MF, self.lams))