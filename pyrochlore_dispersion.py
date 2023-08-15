import numpy as np
import matplotlib.pyplot as plt
import warnings
from sympy.utilities.iterables import multiset_permutations

from numba import njit, jit
from numba.experimental import jitclass
from numba import float32, int16, boolean, complex64
from numba import types, typed, typeof
from misc_helper import *



# def genBZ( d):
#     d = d * 1j
#     b = np.mgrid[-1: 1:d, -1: 1:d, -1: 1:d].reshape(3, -1).T
#     BZ = np.zeros((len(b), 3), dtype=np.single)
#     for i in range (len(b)):
#         BZ[i] = b[i, 0] *BasisBZ(0) + b[i, 1] *BasisBZ(1) + b[i, 2] *BasisBZ(2)
#     return BZ


formfactor = np.zeros((4,4,3))
for i in range(4):
    for j in range(4):
        formfactor[i,j,:] = NN(i)-NN(j)


NNN = np.array([[-1/4, -1/4, -1/4],[-1/4, 1/4, 1/4], [1/4, -1/4, 1/4], [1/4, 1/4, -1/4]])
ZZZ = np.array([[-1, -1, -1],[-1, 1, 1], [1, -1, 1], [1, 1, -1]])/np.sqrt(3)


def exponent_mag(h, n, k, alpha):
    temp = 0
    for i in range(4):
        temp += -1/4 * h * np.dot(n, z(i)) * np.exp(1j*np.dot(k, neta(alpha)*NNtest(i)))
    return temp


def M_zero(Jpm, eta, k, alpha):
    temp = -Jpm/4 *eta[alpha]* np.exp(-1j*neta(alpha)*(np.einsum('ik, jlk->ijl',k, formfactor)))
    temp = np.einsum('ijk->i', temp)
    return temp

def M_zero_single(Jpm, eta, k, alpha):
    temp = -Jpm/4 *eta[alpha]* np.exp(-1j*neta(alpha)*(np.einsum('k, jlk->jl',k, formfactor)))
    temp = np.sum(temp)
    return temp

def M_true(k, Jpm, eta, h, n):
    M = np.zeros((len(k), 2,2), dtype=complex)
    M[:, 0, 0] = M_zero(Jpm, eta, k, 0)
    M[:, 1, 1] = M_zero(Jpm, eta, k, 1)
    M[:, 0, 1] = exponent_mag(h, n, k, 0)
    M[:, 1, 0] = exponent_mag(h, n, k, 1)
    return M


def M_single(lams, k, Jzz, Jpm, eta, h, n):
    M = np.zeros((2,2), dtype=complex)
    M[0, 0] = M_zero_single(Jpm, eta, k, 0)
    M[1, 1] = M_zero_single(Jpm, eta, k, 1)
    M[0, 1] = exponent_mag(h, n, k, 0)
    M[1, 0] = exponent_mag(h, n, k, 1)
    M = M + np.diag(lams)
    E, V = np.linalg.eig(M)
    return np.sqrt(2*Jzz*np.real(E))


def E_zero_true(lams, k, Jpm, eta, h, n):
    M = M_true(k, Jpm, eta, h, n)
    M = M + np.diag(lams)
    E, V = np.linalg.eigh(M)
    return [E,V]

def E_zero_old(lams, k, alpha, Jzz, Jpm, eta):
    return np.sqrt(2*Jzz*(lams[alpha]+M_zero(Jpm, eta, k, alpha)))
def green_f(M, lams, omega):
    temp = M + np.diag(lams) + np.diag(omega**2*np.ones(2)/2)
    return np.linalg.inv(temp)

def green_ff(k, lams, omega, Jpm, eta, h, n):
    M = M_true(k, Jpm, eta, h, n)
    temp = M + np.diag(lams) + np.diag(omega**2*np.ones(2)/2)
    return np.linalg.inv(temp)

def rho_true(M, lams, Jzz):
    temp = M + np.diag(lams)
    E,V = np.linalg.eigh(temp)
    Vt = np.einsum('ijk,ikj->ikj',np.transpose(np.conj(V), (0,2,1)),V)
    Ep = np.real(np.mean(np.einsum('ijk, ik->ij', Vt, Jzz/np.sqrt(2*Jzz*E)), axis=0))
    return Ep


def findLambda_zero(M, Jzz, kappa, tol):
    warnings.filterwarnings("error")
    lamMin = np.zeros(2)
    lamMax = 50*np.ones(2)
    lams = (lamMin + lamMax) / 2
    rhoguess = rho_true(M, lams, Jzz)
    # print(self.kappa)
    yes = True
    while yes >= tol:
        # for i in range(2):
         lams = (lamMin+lamMax)/2
         # rhoguess = rho_true(Jzz, M, lams)
         try:
             rhoguess = rho_true(M, lams, Jzz)
             # rhoguess = self.rho_zero(alpha, self.lams)
             if rhoguess[0] - kappa > 0:
                 lamMin = lams
             else:
                 lamMax = lams
             # if rhoguess[0] - kappa > 0:
             #     lamMin[1] = lams[1]
             # else:
             #     lamMax[1] = lams[1]
         except:
             # print(e)
             lamMin = lams
         # print([lams, rhoguess, np.absolute(rhoguess-kappa)])
         if np.absolute(rhoguess[0]-kappa)<=tol and np.absolute(rhoguess[1]-kappa)<=tol:
             yes = False

    return lams


#graphing BZ

def dispersion_zero(lams, k, Jzz, Jpm, eta, h, n):
    temp = np.sqrt(2*Jzz*E_zero_true(lams, k, Jpm, eta, h, n)[0])
    return temp

def algebraicE(lams,k, h):
    q1 = k[:, 0]
    q2 = k[:, 1]
    q3 = k[:, 2]
    lamA, lamB = lams
    E = np.empty((len(k),2), dtype=complex)

    E[:, 0] = -(1 / 24) * np.exp(-(1 / 4)
                                * 1j * (2 * q1 + 3 * (q2 + q3))) * np.sqrt(np.exp(1 / 2 * 1j * (q1 + 2 * (q2 + q3))) * (
        -12 * np.exp((1j * q1) / 2) * h ** 2 - 12 * np.exp((1j * q2) / 2) * h ** 2 + 4 * np.exp(
            1 / 2 * 1j * (2 * q1 + q2)) * h ** 2 + 4 * np.exp(1 / 2 * 1j * (q1 + 2 * q2)) * h ** 2 - 12 * np.exp(
            (1j * q3) / 2) * h ** 2 + 4 * np.exp(1 / 2 * 1j * (2 * q1 + q3)) * h ** 2 + 4 * np.exp(
            1 / 2 * 1j * (2 * q2 + q3)) * h ** 2 -
        12 * np.exp(1 / 2 * 1j * (2 * q1 + 2 * q2 + q3)) * h ** 2 + 4 * np.exp(
            1 / 2 * 1j * (q1 + 2 * q3)) * h ** 2 + 4 * np.exp(1 / 2 * 1j * (q2 + 2 * q3)) * h ** 2 -
        12 * np.exp(1 / 2 * 1j * (2 * q1 + q2 + 2 * q3)) * h ** 2 - 12 * np.exp(
            1 / 2 * 1j * (q1 + 2 * (q2 + q3))) * h ** 2 + 3 * np.exp(1 / 2 * 1j * (q1 + q2 + q3)) * (
                    16 * h ** 2 + 3 * (lamA - lamB) ** 2))) + (lamA + lamB) / 2

    E[:, 1] = (1 / 24) * np.exp(-(1 / 4)
                                * 1j * (2 * q1 + 3 * (q2 + q3))) *np.sqrt(np.exp(1 / 2 * 1j * (q1 + 2 * (q2 + q3)))*(
        -12 * np.exp((1j * q1) / 2) * h ** 2 - 12 * np.exp((1j * q2) / 2) * h ** 2 + 4 * np.exp(
            1 / 2 * 1j * (2 * q1 + q2)) * h ** 2 + 4 * np.exp(1 / 2 * 1j * (q1 + 2 * q2)) * h ** 2 - 12 * np.exp(
            (1j * q3) / 2) * h ** 2 + 4 * np.exp(1 / 2 * 1j * (2 * q1 + q3)) * h ** 2 + 4 * np.exp(
            1 / 2 * 1j * (2 * q2 + q3)) * h ** 2 -
        12 * np.exp(1 / 2 * 1j * (2 * q1 + 2 * q2 + q3)) * h ** 2 + 4 * np.exp(
            1 / 2 * 1j * (q1 + 2 * q3)) * h ** 2 + 4 * np.exp(1 / 2 * 1j * (q2 + 2 * q3)) * h ** 2 -
        12 * np.exp(1 / 2 * 1j * (2 * q1 + q2 + 2 * q3)) * h ** 2 - 12 * np.exp(
            1 / 2 * 1j * (q1 + 2 * (q2 + q3))) * h ** 2 + 3 * np.exp(1 / 2 * 1j * (q1 + q2 + q3)) * (
                    16 * h ** 2 + 3 * (lamA - lamB) ** 2))) + (lamA + lamB) / 2
    return E

def algdispersion(lams,k,Jzz, h):
    return np.sqrt(2*Jzz*np.real(algebraicE(lams,k, h)))

# def populate(res):
#     temp = np.zeros((1,3))
#     for i in symK:
#         for j in symK:
#             if not (i == j).all():
#                 temp = np.concatenate((temp, np.linspace(i, j, res)))
#             else:
#                 temp = np.concatenate((temp, np.linspace(i, j, 1)))
#     return temp
#
# symK = populate(3)

def calAlgDispersion(lams,Jzz , h):

    dGammaX= algdispersion(lams,GammaX,Jzz, h)
    dXW= algdispersion(lams,XW,Jzz, h)
    dWK = algdispersion(lams,WK,Jzz, h)
    dKGamma = algdispersion(lams,KGamma,Jzz, h)
    dGammaL = algdispersion(lams,GammaL,Jzz, h)
    dLU= algdispersion(lams,LU,Jzz, h)
    dUW = algdispersion(lams,UW,Jzz, h)

    for i in range(2):
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


def calDispersion(lams, Jzz, Jpm, eta, h, n):

    dGammaX= dispersion_zero(lams, GammaX, Jzz, Jpm, eta, h, n)
    dXW= dispersion_zero(lams, XW, Jzz, Jpm, eta, h, n)
    dWK = dispersion_zero(lams, WK, Jzz, Jpm, eta, h, n)
    dKGamma = dispersion_zero(lams, KGamma, Jzz, Jpm, eta, h, n)
    dGammaL = dispersion_zero(lams, GammaL, Jzz, Jpm, eta, h, n)
    dLU= dispersion_zero(lams, LU, Jzz, Jpm, eta, h, n)
    dUW = dispersion_zero(lams, UW, Jzz, Jpm, eta, h, n)

    for i in range(2):
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


def minCal(lams, q, Jzz, Jpm, eta, h, n, K):
    temp = np.zeros((len(q),2))
    mins = np.sqrt(2 * Jzz * E_zero_true(lams, K, Jpm, eta, h, n)[0])
    for i in range(len(q)):
        temp[i] = np.amin(np.sqrt(2 * Jzz * E_zero_true(lams, q[i]+K, Jpm, eta, h, n)[0]) + mins, axis=0)
    return temp

def maxCal(lams, q, Jzz, Jpm, eta, h, n, K):
    temp = np.zeros((len(q),2))
    mins = np.sqrt(2 * Jzz * E_zero_true(lams, K, Jpm, eta, h, n)[0])
    for i in range(len(q)):
        temp[i] = np.amax(np.sqrt(2 * Jzz * E_zero_true(lams, q[i]+K, Jpm, eta, h, n)[0]) + mins, axis=0)
    return temp

def loweredge(lams, Jzz, Jpm, eta, h, n, K):
    dGammaX= minCal(lams, GammaX, Jzz, Jpm, eta, h, n, K)
    dXW= minCal(lams, XW, Jzz, Jpm, eta, h, n, K)
    dWK = minCal(lams, WK, Jzz, Jpm, eta, h, n, K)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, eta, h, n, K)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, eta, h, n, K)
    dLU= minCal(lams, LU, Jzz, Jpm, eta, h, n, K)
    dUW = minCal(lams, UW, Jzz, Jpm, eta, h, n, K)

    for i in range(2):
        plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX[:,i], 'b')
        plt.plot(np.linspace(gX, gW1, len(dXW)), dXW[:, i] , 'b')
        plt.plot(np.linspace(gW1, gK, len(dWK)), dWK[:, i], 'b')
        plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma[:, i], 'b')
        plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL[:, i], 'b')
        plt.plot(np.linspace(gL, gU, len(dLU)), dLU[:, i], 'b')
        plt.plot(np.linspace(gU, gW2, len(dUW)),dUW[:, i], 'b')
    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW2, color='w', label='axvline - full height', linestyle='dashed')
    xlabpos = [gGamma1,gX,gW1,gK,gGamma2,gL,gU,gW2]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)

def upperedge(lams, Jzz, Jpm, eta, h, n, K):
    dGammaX= maxCal(lams, GammaX, Jzz, Jpm, eta, h, n, K)
    dXW= maxCal(lams, XW, Jzz, Jpm, eta, h, n, K)
    dWK = maxCal(lams, WK, Jzz, Jpm, eta, h, n, K)
    dKGamma = maxCal(lams, KGamma, Jzz, Jpm, eta, h, n, K)
    dGammaL = maxCal(lams, GammaL, Jzz, Jpm, eta, h, n, K)
    dLU= maxCal(lams, LU, Jzz, Jpm, eta, h, n, K)
    dUW = maxCal(lams, UW, Jzz, Jpm, eta, h, n, K)

    for i in range(2):
        plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX[:,i], 'b')
        plt.plot(np.linspace(gX, gW1, len(dXW)), dXW[:, i] , 'b')
        plt.plot(np.linspace(gW1, gK, len(dWK)), dWK[:, i], 'b')
        plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma[:, i], 'b')
        plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL[:, i], 'b')
        plt.plot(np.linspace(gL, gU, len(dLU)), dLU[:, i], 'b')
        plt.plot(np.linspace(gU, gW2, len(dUW)),dUW[:, i], 'b')
    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW2, color='w', label='axvline - full height', linestyle='dashed')
    xlabpos = [gGamma1,gX,gW1,gK,gGamma2,gL,gU,gW2]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)



def EMAX(M, lams, Jzz):
    temp = M + np.diag(lams)
    E,V = np.linalg.eigh(temp)
    temp = np.amax(np.sqrt(2*Jzz*E))
    return temp

def GS(lams, k, Jzz, Jpm, eta, h, n):
    return np.mean(dispersion_zero(lams, k, Jzz, Jpm, eta, h, n), axis=0) - lams


class zeroFluxSolver:
    def __init__(self, Jpm, h=0, n=np.array([0,0,0]), eta=1, kappa=2, lam=2, BZres=20, graphres=20, omega=0, Jzz=1):
        self.Jzz = Jzz
        self.Jpm = Jpm
        self.kappa = kappa
        self.eta = np.array([eta, 1], dtype=np.single)
        self.h = h
        self.n = n
        self.omega = omega

        self.tol = 1e-4
        self.lams = np.array([lam, lam], dtype=np.single)

        # self.symK = self.genALLSymPoints()
        # self.symK = self.populate(BZres)

        self.BZres = BZres
        self.graphres = graphres
        self.bigB = np.concatenate((genBZ(BZres), symK))

        self.MF = M_true(self.bigB, self.Jpm, self.eta, self.h, self.n)


    def findLambda(self):
        self.lams = findLambda_zero(self.MF, self.Jzz, self.kappa, self.tol)
        warnings.resetwarnings()


    # def condensed(self):
    #     lams, lmax, lmin = findLambda_zero(self.MF, self.Jzz, self.kappa, self.tol)
    #     if (np.abs(lmax-lmin)<=1e-5).any():
    #         return 1
    #     else:
    #         return 0
    def M_true(self, k):
        return M_true(k, self.Jpm, self.eta, self.h, self.n)

    def LV_zero(self, k):
        return E_zero_true(self.lams, k, self.Jpm, self.eta, self.h, self.n)


    def E_zero(self, k):
        return np.sqrt(2*self.Jzz*E_zero_true(self.lams, k, self.Jpm, self.eta, self.h, self.n)[0])

    def gap(self):
        temp = self.MF + np.diag(self.lams)
        E, V = np.linalg.eigh(temp)
        # E = np.sqrt(2*self.Jzz*E)
        dex = np.argmin(E,axis=0)[0]
        # print("Gap at " + str(self.bigB[dex]) + " with " + str(E[dex, 0]))
        return np.sqrt(2*self.Jzz*E[dex, 0])

    def graph(self, show):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n)
        if show:
            plt.show()

    def graphAlg(self, show):
        calAlgDispersion(self.lams, self.Jzz, self.h)
        if show:
            plt.show()

    def E_single(self, k):
        return M_single(self.lams, k, self.Jzz, self.Jpm, self.eta, self.h, self.n)

    def E_zero_old(self, k, alpha):
        return E_zero_old(self.lams, k, alpha, self.Jzz, self.Jpm, self.eta)

    def rho_dev(self):
        return max(rho_true(self.MF, self.lams, self.Jzz) - self.kappa)

    def EMAX(self):
        return EMAX(self.MF, self.lams, self.Jzz)

    def graph_loweredge(self, show):
        loweredge(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB)
        if show:
            plt.show()

    def graph_upperedge(self, show):
        upperedge(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB)
        if show:
            plt.show()