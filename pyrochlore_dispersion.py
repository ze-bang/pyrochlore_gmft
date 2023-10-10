import numpy as np
import matplotlib.pyplot as plt
import warnings
from sympy.utilities.iterables import multiset_permutations
from misc_helper import *
import numba as nb
from opt_einsum import contract


NNN = np.array([[-1/4, -1/4, -1/4],[-1/4, 1/4, 1/4], [1/4, -1/4, 1/4], [1/4, 1/4, -1/4]])
ZZZ = np.array([[-1, -1, -1],[-1, 1, 1], [1, -1, 1], [1, 1, -1]])/np.sqrt(3)

def exponent_mag(h, n, k, alpha):
    zmag = contract('k,ik->i',n,z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j*neta(alpha)*ffact)
    M = contract('ij,j->i', -1/4*h*ffact, zmag)
    return M

def M_zero(Jpm, eta, k, alpha):
    temp = -Jpm/4 *eta[alpha]* np.exp(-1j*neta(alpha)*(contract('ik, jlk->ijl',k, NNminus)))
    temp = contract('jk, ijk->i', notrace, temp)
    return temp

def exponent_mag_single(h, n, k, theta):
    zmag = contract('k,ik->i',n,z)
    ffact = contract('k, jk->j', k, NN)
    ffact = np.exp(1j*ffact)
    M = contract('j,j', -1/4*h*ffact*(np.cos(theta)-1j*np.sin(theta)), zmag)
    return M

def M_zero_sub_interhopping_AB(k, alpha, Jpmpm, xi):
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j*neta(alpha)*ffact)
    beta = 1-alpha
    tempxb = xi[alpha]
    tempxa = xi[beta]
    M1a = contract('jl, ij, l->i', notrace, -Jpmpm/4 * ffact, tempxb)
    M1b = contract('jl, il, j->i', notrace, -Jpmpm/4 * ffact, tempxb)
    M2a = contract('jl, ij, l->i', notrace, -Jpmpm/4 * ffact, np.conj(tempxa))
    M2b = contract('jl, il, j->i', notrace, -Jpmpm/4 * ffact, np.conj(tempxa))
    return M1a + M1b + M2a + M2b

def M_zero_sub_pairing_AA(k, alpha, Jpmpm, chi, chi0):
    d = np.ones(len(k))
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    beta = 1-alpha
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jk, i->i', notrace, Jpmpm / 8 * tempchi * d)
    M2 = contract('jk, ijk->i', notrace, Jpmpm / 8 * tempchi0 * ffact)
    return M1 + M2


def M_zero_single(Jpm, eta, k, alpha):
    temp = -Jpm/4 *eta[alpha]* np.exp(-1j*neta(alpha)*(contract('k, jlk->jl',k, NNminus)))
    temp = np.sum(temp)
    return temp

def M_true(k,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi):

    dummy = np.zeros((len(k)))

    MBk = M_zero(Jpm, eta, k, 0)
    MAk = M_zero(Jpm, eta, k, 1)
    MBnk = M_zero(Jpm, eta, -k, 0)
    MAnk = M_zero(Jpm, eta, -k, 1)

    MagAkBk = exponent_mag(h, n, k, theta) + M_zero_sub_interhopping_AB(k, 0, Jpmpm, xi)
    MagBkAk = np.conj(MagAkBk)
    MagAnkBnk = exponent_mag(h, n, -k, theta) + M_zero_sub_interhopping_AB(-k, 0, Jpmpm, xi)
    MagBnkAnk = np.conj(MagAnkBnk)

    MAdkAdnk = M_zero_sub_pairing_AA(k, 0, Jpmpm, chi, chi0)
    MBdkBdnk = M_zero_sub_pairing_AA(k, 1, Jpmpm, chi, chi0)
    MAnkAk = np.conj(MAdkAdnk)
    MBnkBk = np.conj(MBdkBdnk)
    #
    # FM = np.block([[MAk, MagAkBk, MAdkAdnk, dummy],
    #                [MagBkAk, MBk, dummy, MBdkBdnk],
    #                [MAnkAk, dummy, MAnk, MagAnkBnk],
    #                [dummy, MBnkBk, MagBnkAnk, MBnk]])
    FM = np.zeros((len(k),4,4), dtype=np.complex128)
    FM[:, 0, 0] = MAk
    FM[:, 0, 1] = MagAkBk
    FM[:, 0, 2] = MAdkAdnk
    FM[:, 0, 3] = dummy
    FM[:, 1, 0] = MagBkAk
    FM[:, 1, 1] = MBk
    FM[:, 1, 2] = dummy
    FM[:, 1, 3] = MBdkBdnk
    FM[:, 2, 0] = MAnkAk
    FM[:, 2, 1] = dummy
    FM[:, 2, 2] = MAnk
    FM[:, 2, 3] = MagAnkBnk
    FM[:, 3, 0] = dummy
    FM[:, 3, 1] = MBnkBk
    FM[:, 3, 2] = MagBnkAnk
    FM[:, 3, 3] = MBnk

    return FM



def M_single(k, Jpm, eta, h, n):
    M = np.zeros((2,2),  dtype=np.complex128)
    M[0, 0] = M_zero_single(Jpm, eta, k, 0)
    M[1, 1] = M_zero_single(Jpm, eta, k, 1)
    M[0, 1] = exponent_mag_single(h, n, k, 0)
    M[1, 0] = exponent_mag_single(h, n, k, 1)
    return M

def E_zero_true(lams, k,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    M = M_true(k,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
    M = M + np.diag(np.repeat(lams,2))
    E, V = np.linalg.eigh(M)
    return [E,V]

def E_zero_fixed(lams, M):
    M = M + np.diag(np.repeat(lams,2))
    E, V = np.linalg.eigh(M)
    return [E,V]
def E_zero_single(lams, k, Jpm, eta, h, n):
    M = M_single(k, Jpm, eta, h, n)
    M = M + np.diag(lams)
    E, V = np.linalg.eigh(M)
    return [E,V]

def E_zero_old(lams, k, alpha, Jzz, Jpm, eta):
    return np.sqrt(2*Jzz*(lams[alpha]+M_zero(Jpm, eta, k, alpha)))

def green_f(M, lams, omega):
    temp = M + np.diag(lams) + np.diag(omega**2*np.ones(2)/2)
    return np.linalg.inv(temp)

# def green_ff(k, lams, omega, Jpm, eta, h, n):
#     M = M_true(k, Jpm, eta, h, n)
#     temp = M + np.diag(lams) + np.diag(omega**2*np.ones(2)/2)
#     return np.linalg.inv(temp)

def rho_true(M, lams, Jzz):
    temp = M + np.diag(np.repeat(lams,2))
    E,V = np.linalg.eigh(temp)
    Vt = np.real(contract('ijk,ijk->ijk',V, np.conj(V)))
    Ep = contract('ijk, ik->ij', Vt, Jzz/np.sqrt(2*Jzz*E))
    return np.mean(Ep)*np.ones(2)


def Emin(k, lams, eta, Jpm, h, n):
    return E_zero_single(lams, k, Jpm, eta, h, n)[0][0]


def gradient(k, lams, eta, Jpm, h, n):
    kx, ky, kz = k
    step = 1e-8
    fx = (Emin(np.array([kx+step, ky, kz]), lams, eta, Jpm, h, n) - Emin(np.array([kx, ky, kz]), lams, eta, Jpm, h, n))/step
    fy = (Emin(np.array([kx, ky+step, kz]), lams, eta, Jpm, h, n) - Emin(np.array([kx, ky, kz]), lams, eta, Jpm, h, n)) / step
    fz = (Emin(np.array([kx, ky, kz+step]), lams, eta, Jpm, h, n) - Emin(np.array([kx, ky, kz]), lams, eta, Jpm, h, n)) / step
    return np.array([fx, fy, fz])



def findminLam_old(M, Jzz, tol):
    warnings.filterwarnings("error")
    lamMin = np.zeros(2)
    lamMax = 50*np.ones(2)
    lams = (lamMin + lamMax) / 2
    while not ((lamMax-lamMin<=tol).all()):
        lams = (lamMin + lamMax) / 2
        try:
             rhoguess = rho_true(M, lams, Jzz)
             for i in range(2):
                 lamMax[i] = lams[i]
        except:
             lamMin = lams
        # print([lams, lamMin, lamMax,lamMax-lamMin])
    warnings.resetwarnings()
    return lams


def findminLam(M, K, tol, eta, Jpm, h, n):
    warnings.filterwarnings("error")

    E, V = np.linalg.eigh(M)
    dex = np.argmin(E[0], axis=0)
    Know = K[dex]
    Enow = 1000
    Enext = E[dex, 0]
    step = 1e-2
    init = True
    while(abs(Enow-Enext)>=tol):
        if not init:
            gradlen = gradient(Know, np.zeros(2), eta, Jpm, h, n)-gradient(Klast, np.zeros(2), eta, Jpm, h, n)
            try:
                step = abs(np.dot(Know-Klast, gradlen))/np.linalg.norm(gradlen)**2
            except:
                step = 1e-2
        
        Klast = Know
        Know = Know - step*gradient(Know, np.zeros(2), eta, Jpm, h, n)
        Enow = Enext
        Enext = Emin(Know, np.zeros(2), eta, Jpm, h, n)
        init=False
    warnings.resetwarnings()
    return -Enext

def findLambda_zero(M, Jzz, kappa, tol):
    warnings.filterwarnings("error")
    lamMin = np.zeros(2)
    lamMax = 50*np.ones(2)
    lams = (lamMin + lamMax) / 2
    rhoguess = rho_true(M, lams, Jzz)
    # print(self.kappa)
    # yes = True
    while not ((np.absolute(rhoguess-kappa)<=tol).all()):
        lams = (lamMin + lamMax) / 2
             # rhoguess = rho_true(Jzz, M, lams)
        try:
             rhoguess = rho_true(M, lams, Jzz)
             for i in range(2):
                 # rhoguess = self.rho_zero(alpha, self.lams)
                 if rhoguess[i] - kappa > 0:
                     lamMin[i] = lams[i]
                 else:
                     lamMax[i] = lams[i]
        except:
             # print(e)
             lamMin = lams
        # print([lams, lamMin, lamMax,lamMax-lamMin, rhoguess])

    return lams


#graphing BZ

def dispersion_zero(lams, k, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi):
    temp = np.sqrt(2*Jzz*E_zero_true(lams, k,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])
    return temp

def algebraicE1110Field(lams,k, h):
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
    return np.sqrt(2*Jzz*np.real(algebraicE1110Field(lams,k, h)))

xipicell_zero = np.array([[1,1,1,1],[1,-1,-1,-1]])


def chi(lams, M, K, Jzz):
    E, V = E_zero_fixed(lams, M)
    E = np.sqrt(2*Jzz*E)
    green = green_zero(E, V, Jzz)
    ffact = contract('ik,jlk->ijl', K, NNminus)
    ffactB = np.exp(-1j * ffact)

    M1 = np.mean(contract('i, jl, ijl->i', green[:,2,0], notrace, ffactB), axis=0)
    return np.real(np.array([M1, M1]))

def chi0(lams, M, Jzz):
    E, V = E_zero_fixed(lams, M)
    E = np.sqrt(2*Jzz*E)
    green = green_zero(E, V, Jzz)

    chi0A = np.mean(green[:, 0, 2])
    chi0B = np.mean(green[:, 1, 3])

    return np.real(np.array([chi0A, chi0B]))

def xi(lams, M, K, Jzz, ns):
    E, V = E_zero_fixed(lams, M)
    E = np.sqrt(2*Jzz*E)
    green = green_zero(E, V, Jzz)
    ffact = contract('ik,jk->ij', K, NN)
    ffactA = np.exp(1j * ffact)

    M1 = np.mean(contract('i, ij->i', green[:,0,1], ffactA), axis=0)
    M1 = M1*xipicell_zero[ns]
    M2 = np.transpose(np.conj(M1))
    return np.real(np.array([M1,M2]))



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


def calDispersion(lams, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi):

    dGammaX= dispersion_zero(lams, GammaX, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi)
    dXW= dispersion_zero(lams, XW, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi)
    dWK = dispersion_zero(lams, WK, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi)
    dKGamma = dispersion_zero(lams, KGamma, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi)
    dGammaL = dispersion_zero(lams, GammaL, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi)
    dLU= dispersion_zero(lams, LU, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi)
    dUW = dispersion_zero(lams, UW, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi)

    for i in range(4):
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


def minCal(lams, q, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K):
    temp = np.zeros(len(q))
    mins = np.sqrt(2 * Jzz * E_zero_true(lams, K,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])[:,0]
    for i in range(len(q)):
        temp[i] = np.min(np.sqrt(2 * Jzz * E_zero_true(lams, K-q[i],eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])[:,0] + mins)
    return temp

def maxCal(lams, q, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K):
    temp = np.zeros(len(q))
    maxs = np.sqrt(2 * Jzz * E_zero_true(lams, K,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])[:,-1]
    for i in range(len(q)):
        temp[i] = np.max(np.sqrt(2 * Jzz * E_zero_true(lams, K-q[i],eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])[:,-1] + maxs)
    return temp

def minMaxCal(lams, q, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K):
    temp = np.zeros((len(q),2))
    maxs = np.sqrt(2 * Jzz * E_zero_true(lams, K,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])
    for i in range(len(q)):
        stuff = np.sqrt(2 * Jzz * E_zero_true(lams, K-q[i],eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0]) + maxs
        temp[i,0] = np.min(stuff)
        temp[i,1] = np.max(stuff)
    return temp


def loweredge(lams, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K):
    dGammaX= minCal(lams, GammaX, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dXW= minCal(lams, XW, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dWK = minCal(lams, WK, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dLU= minCal(lams, LU, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dUW = minCal(lams, UW, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)

    plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, 'b')
    plt.plot(np.linspace(gX, gW1, len(dXW)), dXW, 'b')
    plt.plot(np.linspace(gW1, gK, len(dWK)), dWK, 'b')
    plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, 'b')
    plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, 'b')
    plt.plot(np.linspace(gL, gU, len(dLU)), dLU, 'b')
    plt.plot(np.linspace(gU, gW2, len(dUW)),dUW, 'b')

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

def upperedge(lams, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K):
    dGammaX= maxCal(lams, GammaX, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dXW= maxCal(lams, XW, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dWK = maxCal(lams, WK, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dKGamma = maxCal(lams, KGamma, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dGammaL = maxCal(lams, GammaL, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dLU= maxCal(lams, LU, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)
    dUW = maxCal(lams, UW, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, K)

    plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, 'b')
    plt.plot(np.linspace(gX, gW1, len(dXW)), dXW, 'b')
    plt.plot(np.linspace(gW1, gK, len(dWK)), dWK, 'b')
    plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, 'b')
    plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, 'b')
    plt.plot(np.linspace(gL, gU, len(dLU)), dLU, 'b')
    plt.plot(np.linspace(gU, gW2, len(dUW)),dUW, 'b')

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
    temp = M + np.diag(np.repeat(lams,2))
    E,V = np.linalg.eigh(temp)
    temp = np.amax(np.sqrt(2*Jzz*E))
    return temp

def GS(lams, k, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi):
    return np.mean(dispersion_zero(lams, k, Jzz ,Jpm, Jpmpm,eta, h, n, theta, chi, chi0, xi), axis=0) - lams

def green_zero(E, V, Jzz):
    Vt = contract('ijk,ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    green = Jzz/E
    green = contract('ikjl, ik->ijl', Vt, green)
    return green

def green_zero_branch(E, V, Jzz):
    Vt = contract('ijk,ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    green = Jzz/E
    green = contract('ikjl, ik->ikjl', Vt, green)
    return green



class zeroFluxSolver:
    def __init__(self, Jxx, Jyy, Jzz, h=0, n=np.array([0,0,0]), eta=1, kappa=2, lam=2, BZres=20, graphres=20, omega=0, theta=0, ns=0):
        self.Jzz = Jzz
        self.Jpm = -(Jxx+Jyy)/4
        self.Jpmpm = (Jxx-Jyy)/4
        self.kappa = kappa
        self.eta = np.array([eta, 1], dtype=np.single)
        self.h = h
        self.n = n
        self.omega = omega
        self.theta = theta

        self.tol = 1e-4
        self.lams = np.array([lam, lam], dtype=np.single)
        self.minLams = np.zeros(2)
        # self.symK = self.genALLSymPoints()
        # self.symK = self.populate(BZres)

        self.chi = np.ones(2)
        self.xi = np.array([xipicell_zero[ns], xipicell_zero[ns]])
        self.chi0 = np.ones(2)

        self.BZres = BZres
        self.graphres = graphres
        self.bigB = np.concatenate((genBZ(BZres), symK))

        self.MF = M_true(self.bigB, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)
        self.q = np.empty((len(self.bigB), 3))
        self.q[:] = np.nan


    def findLambda(self):
        self.lams = findLambda_zero(self.MF, self.Jzz, self.kappa, self.tol)
        warnings.resetwarnings()

    def findminLam(self):
        # self.minLams = np.ones(2)*findminLam(self.MF, self.bigB, self.tol, self.eta, self.Jpm, self.h, self.n)
        self.minLams = findminLam_old(self.MF, self.Jzz, 1e-10)

    def solvemeanfield(self, tol=1e-10, ns=0):
        self.findLambda()
        chinext = chi(self.lams, self.MF, self.bigB, self.Jzz)
        xinext = xi(self.lams, self.MF, self.bigB, self.Jzz, ns)
        chi0next = chi0(self.lams, self.MF, self.Jzz)
        while((abs(chinext-self.chi)>=tol).any() or (abs(xinext-self.xi)>=tol).any() or (abs(chi0next-self.chi0)>=tol).any() ):
            self.chi = chinext
            self.chi0 = chi0next
            self.xi = xinext
            self.MF = M_true(self.bigB, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)
            self.findLambda()
            chinext = chi(self.lams, self.MF, self.bigB, self.Jzz)
            chi0next = chi0(self.lams, self.MF, self.Jzz)
            xinext = xi(self.lams, self.MF, self.bigB, self.Jzz, ns)
            print(self.chi, self.chi0, self.xi)
        self.chi = chinext
        self.chi0 = chi0next
        self.xi = xinext
        return 0

    def qvec(self):
        E = E_zero_true(self.lams-np.ones(2)*(1e2/len(self.bigB))**2, self.bigB, self.Jpm, self.eta, self.h, self.n)[0]
        c = np.unique(np.where(E < 0)[0])
        temp = np.unique(self.bigB[c], axis=0)
        self.q[0:len(temp), :] = temp

    def ifcondense(self, q):
        E = E_zero_true(self.lams-np.ones(2)*(1e2/len(self.bigB))**2, q, self.Jpm, self.eta, self.h, self.n)[0]
        c = np.unique(np.where(E < 0)[0])
        return c

    def condensed(self):
        return np.absolute(self.minLams - self.lams) < (1e2/len(self.bigB))**2

    def M_true(self, k):
        return M_true(k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)

    def LV_zero(self, k, lam=np.zeros(2)):
        if np.any(lam == 0):
            lam = self.lams
        return E_zero_true(lam, k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)


    def E_zero(self, k):
        return np.sqrt(2*self.Jzz*E_zero_true(self.lams, k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)[0])

    def gap(self):
        temp = self.MF + np.diag(np.repeat(self.lams,2))
        E, V = np.linalg.eigh(temp)
        dex = np.argmin(E,axis=0)[0]
        return np.sqrt(2*self.Jzz*E[dex, 0])

    def GS(self):
        return np.mean(self.E_zero(self.bigB)) - self.lams[0]


    def gapwhere(self):
        temp = self.MF + np.diag(np.repeat(self.lams,2))
        E, V = np.linalg.eigh(temp)
        # E = np.sqrt(2*self.Jzz*E)
        dex = np.argmin(E,axis=0)[0]
        return np.mod(self.bigB[dex], 2*np.pi)

    def graph(self, show):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)
        if show:
            plt.show()

    def rho(self, lams):
        return rho_true(self.MF, lams, self.Jzz)


    def graphAlg(self, show):
        calAlgDispersion(self.lams, self.Jzz, self.h)
        if show:
            plt.show()

    # def E_single(self, k):
    #     return M_single(self.lams, k, self.Jzz, self.Jpm, self.eta, self.h, self.n)

    def E_zero_old(self, k, alpha):
        return E_zero_old(self.lams, k, alpha, self.Jzz, self.Jpm, self.eta)

    def rho_dev(self):
        return max(rho_true(self.MF, self.lams, self.Jzz) - self.kappa)

    def EMAX(self):
        return EMAX(self.MF, self.lams, self.Jzz)

    def graph_loweredge(self, show):
        loweredge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.bigB)
        if show:
            plt.show()

    def graph_upperedge(self, show):
        upperedge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.bigB)
        if show:
            plt.show()

    # def minCal(self, K):
    #     return minCal(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, K)
    #
    # def maxCal(self, K):
    #     return maxCal(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, K)
    #
    # def minMaxCal(self, K):
    #     return minMaxCal(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, K)

    def TWOSPINON_GAP(self, k):
        return np.min(minCal(self.lams, k, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.bigB))

    def TWOSPINON_MAX(self, k):
        return np.max(maxCal(self.lams, k, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.bigB))

    def green_zero(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_zero(E, V, self.Jzz)

    def green_zero_branch(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_zero_branch(E, V, self.Jzz), E


    def magnetization(self):
        green = self.green_zero(self.bigB)
        zmag = contract('k,ik->i',self.n,z)
        ffact = contract('ik, jk->ij', k, NN)
        ffactp = np.exp(1j*ffact)
        ffactm = np.exp(-1j * ffact)

        magp = contract('j, ij, i->i', zmag, ffactp, green[:,0,1]) / 4
        magm = contract('j, ij, i->i', zmag, ffactm, green[:,1,0]) / 4

        return np.real(np.mean(magp + magm))