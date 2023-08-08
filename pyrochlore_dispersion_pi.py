import numpy as np
import matplotlib.pyplot as plt
import warnings
from numba import jit
from numba.experimental import jitclass
from misc_helper import *


def exponent_pi(k, alpha, mu, nu, rs1, rs2):
    rs = rs1 - neta(alpha) * step(mu)

    f = np.real(np.exp(1j * neta(alpha) * (A_pi(rs, rs2) - A_pi(rs, rs1))))

    return f * np.exp(1j * neta(alpha) * (np.dot(k, NNtest(nu) - NNtest(mu))))


def M_pi_term(k, alpha, rs1, rs2, mu, nu, eta, Jpm):
    temp = -Jpm * eta[alpha] / 4 * exponent_pi(k, alpha, mu, nu, rs1, rs2)
    return temp


def M_pi_mag_term(k, alpha, rs1, mu, h, n):
    rs = rs1 - neta(alpha) * step(mu)
    temp = 1 / 2 * h * np.dot(n, z(mu)) * np.exp(1j * A_pi(rs, rs1)) * np.exp(
        1j * np.dot(k, neta(alpha) * NNtest(mu)))
    return temp


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
                # M[:,i,index2] += -self.M_pi_mag_term(k, alpha, rs2, j)
                # M[:, index2, i] += -np.conj(self.M_pi_mag_term(k, 1-alpha, rs2, j))

    return M


def M_pi(k,eta,Jpm):
    bigM1 = np.zeros((len(k), 4, 4, 4), dtype=complex)
    bigM2 = np.zeros((len(k), 4, 4, 4), dtype=complex)
    for i in range(4):
        bigM1[:, i, :, :] = M_pi_sub(k, i, 0,eta,Jpm)
        bigM2[:, i, :, :] = M_pi_sub(k, i, 1,eta,Jpm)
    M = np.einsum('ijkl->ikl', bigM1)
    M1 = np.einsum('ijkl->ikl', bigM2)
    FM = np.block([[M, np.zeros((len(k), 4, 4))], [np.zeros((len(k), 4, 4)), M1]])
    return FM


def M_pi_sub_single(k, rs, alpha, eta, Jpm):
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
                # M[:,i,index2] += -self.M_pi_mag_term(k, alpha, rs2, j)
                # M[:, index2, i] += -np.conj(self.M_pi_mag_term(k, 1-alpha, rs2, j))

    return M


def M_pi_single(k, eta, Jpm):
    bigM = np.zeros((4, 4, 4), dtype=complex)
    bigM2 = np.zeros((4, 4, 4), dtype=complex)
    for i in range(4):
        bigM[i, :, :] = M_pi_sub_single(k, i, 0, eta, Jpm)
        bigM2[i, :, :] = M_pi_sub_single(k, i, 1, eta, Jpm)
    M = np.einsum('ijk->jk', bigM)
    M1 = np.einsum('ijk->jk', bigM2)
    FM = np.block([[M, np.zeros((4, 4))], [np.zeros((4, 4)), M1]])
    return FM

def E_pi_fixed(lams, M):
    M = M + np.diag(np.repeat(lams,4))
    E, V = np.linalg.eigh(M)
    return [E, V]


def E_pi(lams, k, eta, Jpm):
    M = M_pi(k,eta,Jpm)
    M = M + np.diag(np.repeat(lams,4))
    E, V = np.linalg.eigh(M)
    return [E,V]


def rho_true(Jzz, M, lams):
    dumb = np.array([[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1]])
    temp = M + np.diag(np.repeat(lams,4))
    E,V = np.linalg.eigh(temp)
    Vt = np.real(np.einsum('ijk,ijk->ijk',V, np.conj(V)))
    Ep = np.mean(np.einsum('ijk, ik->ij', Vt, 1/np.sqrt(2*Jzz*E)), axis=0)
    Ep = np.einsum('jk, ik->ij', dumb, Ep)
    return Ep




def findlambda_pi(M, Jzz, kappa, tol):
    warnings.filterwarnings("error")
    lamMin = np.zeros(2)
    lamMax = 10*np.ones(2)
    lams = (lamMin + lamMax) / 2
    rhoguess = rho_true(Jzz, M, lams)
    # print(self.kappa)
    for i in range(2):
        while np.absolute(rhoguess[i]-kappa) >= tol:
             lams[i] = (lamMin[i]+lamMax[i])/2
             # rhoguess = rho_true(Jzz, M, lams)
             try:
                 rhoguess = rho_true(Jzz, M, lams)
                 # rhoguess = self.rho_zero(alpha, self.lams)
                 if rhoguess[i] - kappa > 0:
                     lamMin[i] = lams[i]
                 else:
                     lamMax[i] = lams[i]
             except:
                 # print(e)
                 lamMin[i] = lams[i]
             print([lams[i], lamMin[i], lamMax[i], rhoguess[i]])
             # if lamMax == 0:
             #     break
    return lams



# graphing BZ

def dispersion_pi(lams, k, Jzz, Jpm, eta):
    temp = np.sqrt(2*Jzz*E_pi(lams, k, Jpm, eta)[0])
    return temp


def calDispersion(lams, Jzz, Jpm, eta):
    dGammaX= dispersion_pi(lams, GammaX, Jzz, Jpm, eta)
    dXW= dispersion_pi(lams, XW, Jzz, Jpm, eta)
    dWK = dispersion_pi(lams, WK, Jzz, Jpm, eta)
    dKGamma = dispersion_pi(lams, KGamma, Jzz, Jpm, eta)
    dGammaL = dispersion_pi(lams, GammaL, Jzz, Jpm, eta)
    dLU= dispersion_pi(lams, LU, Jzz, Jpm, eta)
    dUW = dispersion_pi(lams, UW, Jzz, Jpm, eta)

    for i in range(8):
        plt.plot(np.linspace(-0.5, 0, graphres), dGammaX[:,i], 'b')
        plt.plot(np.linspace(0, 0.3, graphres), dXW[:, i] , 'b')
        plt.plot(np.linspace(0.3, 0.5, graphres), dWK[:, i], 'b')
        plt.plot(np.linspace(0.5, 0.9, graphres), dKGamma[:, i], 'b')
        plt.plot(np.linspace(0.9, 1.3, graphres), dGammaL[:, i], 'b')
        plt.plot(np.linspace(1.3, 1.6, graphres), dLU[:, i], 'b')
        plt.plot(np.linspace(1.6, 1.85, graphres),dUW[:, i], 'b')
    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=-0.5, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.3, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.5, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.9, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.3, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.6, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.85, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [-0.5,0,0.3,0.5,0.9,1.3,1.6,1.85]
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
#     for i in range(2):
#         plt.plot(np.linspace(-0.5, 0, graphres), dGammaX[:,i], 'w')
#         plt.plot(np.linspace(0, 0.3, graphres), dXW[:, i] , 'w')
#         plt.plot(np.linspace(0.3, 0.5, graphres), dWK[:, i], 'w')
#         plt.plot(np.linspace(0.5, 0.9, graphres), dKGamma[:, i], 'w')
#         plt.plot(np.linspace(0.9, 1.3, graphres), dGammaL[:, i], 'w')
#         plt.plot(np.linspace(1.3, 1.6, graphres), dLU[:, i], 'w')
#         plt.plot(np.linspace(1.6, 1.85, graphres),dUW[:, i], 'w')
#     plt.ylabel(r'$\omega/J_{zz}$')
#     plt.axvline(x=-0.5, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=0, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=0.3, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=0.5, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=0.9, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=1.3, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=1.6, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=1.85, color='w', label='axvline - full height', linestyle='dashed')
#     xlabpos = [-0.5,0,0.3,0.5,0.9,1.3,1.6,1.85]
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
#     for i in range(2):
#         plt.plot(np.linspace(-0.5, 0, graphres), dGammaX[:,i], 'w')
#         plt.plot(np.linspace(0, 0.3, graphres), dXW[:, i] , 'w')
#         plt.plot(np.linspace(0.3, 0.5, graphres), dWK[:, i], 'w')
#         plt.plot(np.linspace(0.5, 0.9, graphres), dKGamma[:, i], 'w')
#         plt.plot(np.linspace(0.9, 1.3, graphres), dGammaL[:, i], 'w')
#         plt.plot(np.linspace(1.3, 1.6, graphres), dLU[:, i], 'w')
#         plt.plot(np.linspace(1.6, 1.85, graphres),dUW[:, i], 'w')
#     plt.ylabel(r'$\omega/J_{zz}$')
#     plt.axvline(x=-0.5, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=0, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=0.3, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=0.5, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=0.9, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=1.3, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=1.6, color='w', label='axvline - full height', linestyle='dashed')
#     plt.axvline(x=1.85, color='w', label='axvline - full height', linestyle='dashed')
#     xlabpos = [-0.5,0,0.3,0.5,0.9,1.3,1.6,1.85]
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
        self.eta = np.array([eta, 1])
        self.tol = 1e-3
        self.lams =[lam, lam]
        self.h = h
        self.n = n


        self.minLams = np.zeros(2)

        self.BZres = BZres
        self.graphres = graphres
        self.bigB = np.concatenate((genBZ(BZres), symK))


        self.bigK = np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW))
        self.MF = M_pi(self.bigB, self.eta, self.Jpm)


    #alpha = 1 for A = -1 for B


    def findLambda(self):
        self.lams = findlambda_pi(self.MF, self.Jzz, self.kappa, self.tol)

    def M_true(self, k):
        return M_pi(k, self.Jpm, self.eta)


    def E_pi(self, k):
        return np.sqrt(2*self.Jzz*E_pi(self.lams, k, self.Jpm, self.eta)[0])

    def gap(self):
        return np.sqrt(2*self.Jzz*gap(self.MF, self.lams))

    def graph(self, show):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n)
        if show:
            plt.show()

    def E_single(self, k):
        M = M_pi_single(k, self.eta, self.Jpm) + np.diag(np.repeat(self.lams, 4))
        E, V = np.linalg.eigh(M)
        return np.sqrt(2*self.Jzz*E)

    def EMAX(self):
        return np.sqrt(2*self.Jzz*EMAX(self.MF, self.lams))