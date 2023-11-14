import numpy as np
import matplotlib.pyplot as plt
import warnings
from sympy.utilities.iterables import multiset_permutations
from misc_helper import *
import numba as nb
from opt_einsum import contract

xipicell_zero = np.array([[1,1,1,1],[1,-1,-1,-1]])

NNN = np.array([[-1/4, -1/4, -1/4],[-1/4, 1/4, 1/4], [1/4, -1/4, 1/4], [1/4, 1/4, -1/4]])
ZZZ = np.array([[-1, -1, -1],[-1, 1, 1], [1, -1, 1], [1, 1, -1]])/np.sqrt(3)

#region constructing single k point matrix
def exponent_mag_single(h, n, k, theta):
    zmag = contract('k,ik->i',n,z)
    ffact = contract('k, jk->j', k, NN)
    ffact = np.exp(1j*ffact)
    M = contract('j,j->', -1/4*h*ffact*(np.cos(theta)-1j*np.sin(theta)), zmag)
    M = np.real(M)
    return M

def M_zero_single(Jpm, eta, k, alpha):
    temp = -Jpm/4 *eta[alpha]* np.exp(-1j*neta(alpha)*(contract('k, jlk->jl',k, NNminus)))
    temp = contract('jl,jl->', notrace, temp)
    return temp

def M_zero_sub_interhopping_AB_single(k, alpha, Jpmpm, xi):
    ffact = contract('k, jk->j', k, NN)
    ffact = np.exp(1j*neta(alpha)*ffact)
    beta = 1-alpha
    tempxb = xi[alpha]
    tempxa = xi[beta]
    M1a = contract('jl, j, l->', notrace, Jpmpm/4 * ffact, tempxb)
    M1b = contract('jl, l, j->', notrace, Jpmpm/4 * ffact, tempxb)
    M2a = contract('jl, j, l->', notrace, Jpmpm/4 * ffact, np.conj(tempxa))
    M2b = contract('jl, l, j->', notrace, Jpmpm/4 * ffact, np.conj(tempxa))
    return M1a + M1b + M2a + M2b

def M_zero_sub_pairing_AA_single(k, alpha, Jpmpm, chi, chi0):
    ffact = contract('k, jlk->jl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    beta = 1-alpha
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jk, jk->', notrace, Jpmpm / 8 * tempchi)
    M2 = contract('jk, jk->', notrace, Jpmpm / 8 * tempchi0 * ffact)
    return M1 + M2


def M_single(k,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi):

    chi = chi*np.array([notrace, notrace])
    chi0 = chi0*np.ones(2)
    xi = xi*np.array([xipicell_zero[0], xipicell_zero[0]])

    dummy = 0

    MAk = M_zero_single(Jpm, eta, k, 0)
    MBk = M_zero_single(Jpm, eta, k, 1)
    MAnk = M_zero_single(Jpm, eta, -k, 0)
    MBnk = M_zero_single(Jpm, eta, -k, 1)

    MagAkBk = exponent_mag_single(h, n, k, theta) + M_zero_sub_interhopping_AB_single(k, 0, Jpmpm, xi)
    MagBkAk = np.conj(MagAkBk)
    MagAnkBnk = exponent_mag_single(h, n, -k, theta) + M_zero_sub_interhopping_AB_single(-k, 0, Jpmpm, xi)
    MagBnkAnk = np.conj(MagAnkBnk)

    MAdkAdnk = M_zero_sub_pairing_AA_single(k, 0, Jpmpm, chi, chi0)
    MBdkBdnk = M_zero_sub_pairing_AA_single(k, 1, Jpmpm, chi, chi0)
    MAnkAk = np.conj(MAdkAdnk)
    MBnkBk = np.conj(MBdkBdnk)
    #
    # FM = np.block([[MAk, MagAkBk, MAdkAdnk, dummy],
    #                [MagBkAk, MBk, dummy, MBdkBdnk],
    #                [MAnkAk, dummy, MAnk, MagAnkBnk],
    #                [dummy, MBnkBk, MagBnkAnk, MBnk]])
    FM = np.zeros((4,4), dtype=np.complex128)
    FM[0, 0] = MAk
    FM[0, 1] = MagAkBk
    FM[0, 2] = MAdkAdnk
    FM[0, 3] = dummy
    FM[1, 0] = MagBkAk
    FM[1, 1] = MBk
    FM[1, 2] = dummy
    FM[1, 3] = MBdkBdnk
    FM[2, 0] = MAnkAk
    FM[2, 1] = dummy
    FM[2, 2] = MAnk
    FM[2, 3] = MagAnkBnk
    FM[3, 0] = dummy
    FM[3, 1] = MBnkBk
    FM[3, 2] = MagBnkAnk
    FM[3, 3] = MBnk

    return FM

#endregion


#region Constructing Hamiltonian
def exponent_mag(h, n, k, theta):
    zmag = contract('k,ik->i',n,z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j*ffact)
    M = contract('ij,j->i', -1/4*h*ffact*(np.cos(theta) - 1j * np.sin(theta)), zmag)
    return M

def M_zero(Jpm, eta, k, alpha):
    temp = -Jpm/4 *eta[alpha]* np.exp(-1j*neta(alpha)*(contract('ik, jlk->ijl',k, NNminus)))
    temp = contract('jk, ijk->i', notrace, temp)
    return temp

def M_zero_sub_interhopping_AB(k, alpha, Jpmpm, xi):
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j*neta(alpha)*ffact)
    beta = 1-alpha
    tempxb = xi[alpha]
    tempxa = xi[beta]
    M1a = contract('jl, ij, l->i', notrace, Jpmpm/4 * ffact, tempxb)
    M1b = contract('jl, il, j->i', notrace, Jpmpm/4 * ffact, tempxb)
    M2a = contract('jl, ij, l->i', notrace, Jpmpm/4 * ffact, np.conj(tempxa))
    M2b = contract('jl, il, j->i', notrace, Jpmpm/4 * ffact, np.conj(tempxa))
    return M1a + M1b + M2a + M2b

def M_zero_sub_pairing_AA(k, alpha, Jpmpm, chi, chi0):
    d = np.ones(len(k))
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    beta = 1-alpha
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jk, jk, i->i', notrace, Jpmpm / 8 * tempchi, d)
    M2 = contract('jk, ijk->i', notrace, Jpmpm / 8 * tempchi0 * ffact)
    return M1 + M2

def M_true(k,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi):

    dummy = np.zeros((len(k)))
    chi = chi*np.array([notrace, notrace])
    chi0 = chi0*np.ones(2)
    xi = xi*np.array([xipicell_zero[0], xipicell_zero[0]])

    MAk = M_zero(Jpm, eta, k, 0)
    MBk = M_zero(Jpm, eta, k, 1)
    MAnk = M_zero(Jpm, eta, -k, 0)
    MBnk = M_zero(Jpm, eta, -k, 1)

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

#endregion

def E_zero_true(lams, k,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    M = M_true(k,eta,Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
    M = M + np.diag(np.repeat(lams,2))
    E, V = np.linalg.eigh(M)
    return [E,V]

def E_zero_fixed(lams, M):
    M = M + np.diag(np.repeat(lams,2))
    E, V = np.linalg.eigh(M)
    return [E,V]


def E_zero_single(lams, k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    M = M_single(k,eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
    M = M + np.diag(np.repeat(lams,2))
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


def rho_true_site(M, lams, Jzz):
    temp = M + np.diag(np.repeat(lams,2))
    E,V = np.linalg.eigh(temp)
    Vt = np.real(contract('ijk,ijk->ijk',V, np.conj(V)))
    Ep = contract('ijk, ik->ij', Vt, Jzz/np.sqrt(2*Jzz*E))
    return np.mean(Ep, axis=0)[0:2]

def Emin(k, lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    return E_zero_single(lams, k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0][0]


def gradient(k, lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    kx, ky, kz = k
    step = 1e-8
    fx = (Emin(np.array([kx+step, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi) - Emin(np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi))/step
    fy = (Emin(np.array([kx, ky+step, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi) - Emin(np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)) / step
    fz = (Emin(np.array([kx, ky, kz+step]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi) - Emin(np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)) / step
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


def findminLam(M, K, tol, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    warnings.filterwarnings("error")
    E, V = np.linalg.eigh(M)
    E = np.around(E[:,0], decimals=15)
    Em = E.min()
    dex = np.where(E==Em)
    Know = K[dex]

    if Know.shape == (3,):
        Know = Know.reshape(1,3)

    if len(Know) >= 4:
        Know = Know[0:4]

    if (E==0).all():
        return 0, np.array([0,0,0]).reshape((1,3))

    step = 1
    Enow = Em*np.ones(len(Know))
    for i in range(len(Know)):
        stuff = True
        init = True
        while stuff:
            # print(Enow, i, Know[i])
            if not init:
                # print(Enow[i], i, Know[i], gradient(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi),
                      # gradient(Klast, np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi))
                gradlen = gradient(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi) - gradient(Klast, np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
                try:
                    step = abs(np.dot(Know[i] - Klast, gradlen)) / np.linalg.norm(gradlen) ** 2
                except:
                    step = 0

            Klast = np.copy(Know[i])
            Know[i] = Know[i] - step * gradient(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
            Elast = np.copy(Enow[i])
            Enow[i] = Emin(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
            init = False
            if abs(Elast-Enow[i])<1e-12:
                stuff = False
    warnings.resetwarnings()

    Em = min(Enow)
    a = np.where(Enow==Em)
    Know = Know[a]
    Know = np.mod(Know, 2 * np.pi)
    for k in range(len(Know)):
        for i in range(3):
            if Know[k,i] > np.pi:
                Know[k, i] = Know[k, i] - 2*np.pi
    Know = np.unique(Know, axis=0)
    return -Em, Know

def findLambda_zero(M, Jzz, kappa, tol, lamM):
    warnings.filterwarnings("error")
    lamMin = np.max(lamM[0]-1, 0)*np.ones(2)
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
        if (lamMax-lamMin<1e-15).all():
            break
        # print([lams, lamMin, lamMax,lamMax-lamMin, rhoguess])
    warnings.resetwarnings()
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



#region calculating mean field
def chiCal(lams, M, K, Jzz):
    E, V = E_zero_fixed(lams, M)
    E = np.sqrt(2*Jzz*E)
    green = green_zero(E, V, Jzz)
    ffact = contract('ik,jlk->ijl', K, NNminus)
    ffactB = np.exp(1j * ffact)

    M1 = np.mean(contract('i, jl, ijl->ijl', green[:,2,0], notrace, ffactB), axis=0)
    M1 = M1[0,3]
    return M1

def chi0Cal(lams, M, Jzz):
    E, V = E_zero_fixed(lams, M)
    E = np.sqrt(2*Jzz*E)
    green = green_zero(E, V, Jzz)
    chi0A = np.mean(green[:, 0, 2])
    return chi0A

def xiCal(lams, M, K, Jzz, ns):
    E, V = E_zero_fixed(lams, M)
    E = np.sqrt(2*Jzz*E)
    green = green_zero(E, V, Jzz)
    ffact = contract('ik,jk->ij', K, NN)
    ffactA = np.exp(1j * ffact)
    A = contract('i, ij->ij', green[:,0,1], ffactA)
    A = np.mean(A, axis=0)
    return np.real(A[0])

def calmeanfield(lams, M, K, Jzz, ns):
    return chiCal(lams, M, K, Jzz), chi0Cal(lams, M, Jzz), xiCal(lams, M, K, Jzz, ns)

#endregion

#region Calculating condensed mean field

def chiCalC(rhos, K):
    ffact = contract('ik,jlk->ijl', K, NNminus)
    ffactB = np.exp(1j * ffact)
    M1 = np.mean(contract('jl, ijl->ijl', rhos[0] * rhos[0] * notrace, ffactB), axis=0)
    M1 = M1[0,3]
    return M1

def chi0CalC(rhos):
    chi0A = rhos[0] * rhos[0]
    return chi0A

def xiCalC(rhos, K):
    ffact = contract('ik,jk->ij', K, NN)
    ffactA = np.exp(1j * ffact)
    A = rhos[0] * rhos[1] * ffactA
    A = np.mean(A, axis=0)
    return np.real(A[0])

def calmeanfieldC(rhos, K):
    return chiCalC(rhos, K), chi0CalC(rhos), xiCalC(rhos, K)


#endregion

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
    Vt = contract('ijk,ilk->iklj', V, np.conj(V))
    green = Jzz/E
    green = contract('ikjl, ik->ijl', Vt, green)
    return green

def green_zero_branch(E, V, Jzz):
    Vt = contract('ijk,ilk->iklj', V, np.conj(V))
    green = Jzz/E
    green = contract('ikjl, ik->ikjl', Vt, green)
    return green

def MFE(Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, M, lams, k):
    chi = chi * np.array([notrace, notrace])
    chi0 = chi0 * np.ones(2)
    xi = xi * np.array([xipicell_zero[0], xipicell_zero[0]])

    M = M + np.diag(np.repeat(lams,2))

    E, V = np.linalg.eigh(M)
    E = np.sqrt(2*Jzz*E)
    Vt = contract('ijk, ilk->iklj', V, np.conj(V))
    green = green_zero(E, V, Jzz)
    ffact = contract('ik, jlk->ijl', k,NNminus)
    ffactA = np.exp(-1j * ffact)
    ffactB = np.exp(1j * ffact)


    EQ = np.real(np.trace(np.mean(contract('ikjl, ik->ijl', Vt, E/2), axis=0))/2)

    E1A = np.mean(contract('jl, i, ijl->i', notrace, -Jpm/4 * green[:,0,0], ffactA), axis=0)
    E1B = np.mean(contract('jl, i, ijl->i', notrace, -Jpm/4 * green[:,1,1], ffactB), axis=0)

    E1 = np.real(E1A+E1B)

    zmag = contract('k,ik->i',n,z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j*ffact)
    Emag = np.mean(contract('ku, u, k->k',-1/4*h*ffact*(np.cos(theta)-1j*np.sin(theta)), zmag, green[:,0,1]), axis=0)

    Emag = 2*np.real(np.sum(Emag))

    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    tempxb = xi[1]
    tempxa = xi[0]
    M1a = np.mean(contract('jl, ij, l, i->i', notrace, Jpmpm / 4 * ffact, tempxa, green[:,0,1]), axis=0)
    M1b = np.mean(contract('jl, il, j, i->i', notrace, Jpmpm / 4 * ffact, tempxa, green[:,0,1]), axis=0)
    M2a = np.mean(contract('jl, ij, l, i->i', notrace, Jpmpm / 4 * ffact, np.conj(tempxb), green[:,0,1]), axis=0)
    M2b = np.mean(contract('jl, il, j, i->i', notrace, Jpmpm / 4 * ffact, np.conj(tempxb), green[:,0,1]), axis=0)
    EAB = M1a + M1b + M2a + M2b
    EAB = 2 * np.real(EAB)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    beta = 1
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = np.mean(contract('jl, jl, i->i', notrace, Jpmpm / 8 * tempchi, green[:,0,2]), axis=0)
    M2 = np.mean(contract('jl, ijl, i->i', notrace, Jpmpm / 8 * ffact * tempchi0, green[:,0,2]), axis=0)

    EAA = np.real(M1+M2)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * ffact)
    beta = 0
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = np.mean(contract('jl, jl, i->i', notrace, Jpmpm / 8 * tempchi, green[:,1,3]), axis=0)
    M2 = np.mean(contract('jl, ijl, i->i', notrace, Jpmpm / 8 * ffact * tempchi0, green[:,1,3]), axis=0)

    EBB = np.real(M1+M2)

    E = EQ + Emag + E1 + EAB + EAA + EBB
    # print(EQ, E1, Emag, EAB, EAA, EBB)
    return E


# def MFE_condensed(Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, M, lams, q, rho):
#
#     chi = chi * np.array([notrace, notrace])
#     chi0 = chi0 * np.ones(2)
#     xi = xi * np.array([xipicell_zero[0], xipicell_zero[0]])
#
#     k = np.copy(q)
#     k = k.reshape(3,)
#
#     ffact = contract('k, jlk->jl', k, NNminus)
#     ffactA = np.exp(-1j * ffact) + np.exp(1j * ffact)
#
#     E1 = contract('jl, jl->', notrace, -Jpm/4 * ffactA * rho[0] * rho[0])
#
#     zmag = contract('k,ik->i',n,z)
#
#     ffact = contract('k, jk->j', k, NN)
#     ffact = np.exp(1j*ffact)
#
#     Emag = contract('u, u->',-1/4*h*ffact*(np.cos(theta)-1j*np.sin(theta)), rho[0] * rho[1]* zmag)
#
#     Emag = 2*np.real(np.sum(Emag))
#
#     ffact = contract('k, jk->j', k, NN)
#     ffact = np.exp(1j * ffact)
#     tempxb = xi[1]
#     tempxa = xi[0]
#
#     M1a = contract('jl, j, l->', notrace, Jpmpm / 4 * ffact, tempxa * rho[0] * rho[1])
#     M1b = contract('jl, l, j->', notrace, Jpmpm / 4 * ffact, tempxa* rho[0] * rho[1])
#     M2a = contract('jl, j, l->', notrace, Jpmpm / 4 * ffact, np.conj(tempxb)* rho[0] * rho[1])
#     M2b = contract('jl, l, j->', notrace, Jpmpm / 4 * ffact, np.conj(tempxb)* rho[0] * rho[1])
#     EAB = M1a + M1b + M2a + M2b
#     EAB = 2 * np.real(EAB)
#
#     ffact = contract('k, jlk->jl', k, NNminus)
#     ffact = np.exp(-1j * ffact)
#     beta = 1
#     tempchi = chi[beta]
#     tempchi0 = chi0[beta]
#
#     M1 = contract('jl, jl->', notrace, Jpmpm / 8 * tempchi * rho[0] * rho[0])
#     M2 = contract('jl, jl->', notrace, Jpmpm / 8 * ffact * tempchi0 * rho[0] * rho[0])
#
#     EAA = np.real(M1+M2)
#
#     ffact = contract('k, jlk->jl', k, NNminus)
#     ffact = np.exp(1j * ffact)
#     beta = 0
#     tempchi = chi[beta]
#     tempchi0 = chi0[beta]
#
#     M1 = contract('jl, jl->', notrace, Jpmpm / 8 * tempchi * rho[1] * rho[1])
#     M2 = contract('jl, jl->', notrace, Jpmpm / 8 * ffact * tempchi0 * rho[1] * rho[1])
#
#     EBB = np.real(M1+M2)
#
#     # print(E1, Emag, EAB, EAA, EBB)
#     E = Emag + E1 + EAB + EAA + EBB
#     return np.real(E)

def MFE_condensed(Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, M, lams, k, rho):
    chi = chi * np.array([notrace, notrace])
    chi0 = chi0 * np.ones(2)
    xi = xi * np.array([xipicell_zero[0], xipicell_zero[0]])

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffactA = np.exp(-1j * ffact) + np.exp(1j * ffact)

    E1 = contract('jl, ijl->i', notrace, -Jpm/4 * ffactA * rho[0] * rho[0])

    zmag = contract('k,ik->i',n,z)

    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j*ffact)

    Emag = contract('iu, u->i',-1/4*h*ffact*(np.cos(theta)-1j*np.sin(theta)), rho[0] * rho[1]* zmag)

    Emag = 2*np.real(np.mean(Emag))

    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    tempxb = xi[1]
    tempxa = xi[0]

    M1a = contract('jl, ij, l->i', notrace, Jpmpm / 4 * ffact, tempxa * rho[0] * rho[1])
    M1b = contract('jl, il, j->i', notrace, Jpmpm / 4 * ffact, tempxa* rho[0] * rho[1])
    M2a = contract('jl, ij, l->i', notrace, Jpmpm / 4 * ffact, np.conj(tempxb)* rho[0] * rho[1])
    M2b = contract('jl, il, j->i', notrace, Jpmpm / 4 * ffact, np.conj(tempxb)* rho[0] * rho[1])
    EAB = M1a + M1b + M2a + M2b
    EAB = 2 * np.real(np.mean(EAB))

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    beta = 1
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, jl->', notrace, Jpmpm / 8 * tempchi * rho[0] * rho[0])
    M2 = np.mean(contract('jl, ijl->i', notrace, Jpmpm / 8 * ffact * tempchi0 * rho[0] * rho[0]))

    EAA = np.real(M1+M2)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * ffact)
    beta = 0
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, jl->', notrace, Jpmpm / 8 * tempchi * rho[1] * rho[1])
    M2 = np.mean(contract('jl, ijl->i', notrace, Jpmpm / 8 * ffact * tempchi0 * rho[1] * rho[1]))

    EBB = np.real(M1+M2)

    E = np.mean(Emag + E1 + EAB + EAA + EBB)
    return np.real(E)

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

        self.chi = 1+1j
        self.xi = 1
        self.chi0 = 1+1j

        self.delta= np.zeros(2)

        self.BZres = BZres
        self.graphres = graphres
        self.bigB = np.concatenate((genBZ(BZres), symK))
        self.bigB = np.unique(self.bigB, axis=0)
        # self.bigB = genBZ(BZres)
        self.bigTemp = np.copy(self.bigB)
        self.MF = M_true(self.bigB, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)
        self.MForig = np.copy(self.MF)
        self.q = np.empty((len(self.bigB), 3))
        self.q[:] = np.nan
        self.qmin = np.empty(3)
        self.qmin[:] = np.nan
        self.condensed = False
        self.rhos = np.zeros(2)
        self.ns = 0

    def findLambda(self, MF, minLams):
        return findLambda_zero(MF, self.Jzz, self.kappa, self.tol, minLams)

    def findminLam_old(self):
        self.minLams = findminLam_old(self.MF, self.Jzz, 1e-10)

    def findminLam(self, chi, chi0, xi):
        minLams, self.qmin = findminLam(self.MF, self.bigB, self.tol, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, chi0, xi)
        minLams = np.ones(2)*minLams
        K = np.unique(np.concatenate((self.bigB, self.qmin)), axis=0)
        MF = M_true(K, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, chi0, xi)
        return minLams, K, MF

    def calmeanfield(self, lam, MF, K):
        if self.condensed:
            # cond = self.ifcondense(K, lam, (680 / len(K)) ** 2)
            # Kps = np.delete(K, cond, axis=0)
            # MFp = np.delete(MF, cond, axis=0)

            chic, chi0c, xic = calmeanfieldC(self.rhos, self.qmin)
            chi, chi0, xi = calmeanfield(lam, MF, K, self.Jzz, self.ns)
        # print(np.array([chi, chi0, xi]), np.array([chic, chi0c, xic]))
            return chi+chic, chi0+chi0c, xi + xic
        else:
            chi, chi0, xi = calmeanfield(lam, MF, K, self.Jzz, self.ns)
        return np.array([chi, chi0, xi])


    def solvemeanfield(self, tol=0.005, ns=0):
        mfs = np.array([self.chi, self.chi0, self.xi])
        # start = time.time()
        lam, K, MF = self.condensation_check(mfs)
        # end = time.time()
        # print('find min lam and lam routine costs ' + str(end-start))
        # start = time.time()
        mfs = self.calmeanfield(lam, MF, K)
        # end = time.time()
        # print('cal mean field routine costs ' + str(end-start))
        do = not (self.Jpmpm == 0)
        counter = 0
        while do:
            mfslast = np.copy(mfs)
            lam, K, MF = self.condensation_check(mfs)
            mfs = self.calmeanfield(lam, MF, K)
            if (abs(mfs + mfslast) < tol).all() or (abs(mfs - mfslast) < tol).all() or counter >= 10:
                break
            counter = counter + 1
        if do:
            lam, K, MF = self.condensation_check(mfs)
        self.chi, self.chi0, self.xi = mfs
        self.lams = lam
        self.MF = MF
        self.bigTemp = K
        return 0

    def qvec(self):
        E = E_zero_true(self.lams-np.ones(2)*(1e2/len(self.bigB))**2, self.bigB, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)[0]
        c = np.unique(np.where(E < 0)[0])
        temp = np.unique(self.bigB[c], axis=0)
        self.q[0:len(temp), :] = temp

    def ifcondense(self, q, lams, tol=0):
        c = np.array([])
        if self.condensed:
            E, V = self.LV_zero(q, lams)
            E = E[:,0]
            c = np.where(E<=tol)[0]
        c = np.array(c, dtype=int)
        return c

    def set_condensed(self, lams, minLams, l):
        self.condensed = (lams[0]-minLams[0]) < (680/l)**2
        # self.condensed = self.gap(MF, lams) < 0.05574196166518704
        # return np.sqrt(lams[0] - minLams[0])*l

    #0.05574196166518704
    def set_delta(self, K, MF, minLams, lams, l):
        if self.condensed:
            self.delta = np.sqrt(lams-minLams)*len(self.bigTemp)
            cond = self.ifcondense(K, lams, (680/l)**2)
            MFp = np.delete(MF, cond, axis=0)
            warnings.filterwarnings('error')
            try:
                self.rhos = np.sqrt(self.kappa - rho_true_site(MFp, lams, self.Jzz))
            except:
                self.rhos = np.zeros(2)
            warnings.resetwarnings()
    def condensation_check(self, mfs):
        chi, chi0, xi = mfs
        minLams, K, MF = self.findminLam(chi, chi0, xi)
        self.minLams = minLams
        lams = self.findLambda(MF, minLams)
        l = len(K)
        self.set_condensed(lams, minLams, l)
        self.set_delta(K, MF, minLams, lams, l)
        return lams, K, MF

    def M_true(self, k):
        return M_true(k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)

    def LV_zero(self, k, lam=np.zeros(2)):
        if np.any(lam == 0):
            lam = self.lams
        return E_zero_true(lam, k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)


    def E_zero(self, k):
        return np.sqrt(2*self.Jzz*E_zero_true(self.lams, k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)[0])

    def gap(self, MF=0, lams=0):
        if MF == 0:
            MF = self.MF
            lams = self.lams
        temp = MF + np.diag(np.repeat(lams,2))
        E, V = np.linalg.eigh(temp)
        dex = np.argmin(E[:,0],axis=0)
        return np.sqrt(2*self.Jzz*E[dex, 0])

    def GS(self):
        return np.mean(self.E_zero(self.bigB)) - np.mean(self.lams)

    def MFE(self):
        if self.condensed:
            Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.MF,
                     self.lams, self.bigTemp)
            Eq = MFE_condensed(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.MF,
                     self.lams, self.qmin, self.rhos)
            # print(Eq, Ep, self.qmin)
            return Ep + Eq
        else:
            Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.MF,
                self.lams, self.bigTemp)
        return Ep

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

    def mag_con(self):
        return np.mean(E_zero_fixed(np.zeros(2), self.MF)[0])

    def magnetization(self):
        green = self.green_zero(self.bigTemp)
        ffact = contract('ik, jk->ij', self.bigTemp, NN)
        ffactp = np.exp(1j*ffact)
        ffactm = np.exp(-1j * ffact)

        magp = contract('ij, i->i', ffactp, green[:,0,1])
        magm = contract('ij, i->i', ffactm, green[:,1,0])

        con = 0
        # print(self.minLams, self.lams)
        if self.condensed:
            # cond = self.ifcondense(self.bigTemp, self.gap()**2/(2*self.Jzz))
            # Kq = self.bigTemp[cond]

            ffact = contract('ik, jk->j', self.qmin, NN)
            ffactp = np.exp(1j * ffact)
            ffactm = np.exp(-1j * ffact)
            con = self.rhos[0]*self.rhos[1]*np.mean(ffactp+ffactm)
            # con = np.mean(contract('ij->i',ffactp+ffactm))

        return np.real(np.mean(magp + magm)+con)/4