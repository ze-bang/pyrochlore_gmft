import matplotlib.pyplot as plt
import warnings

import numpy as np

from misc_helper import *

# A_pi = np.array([[0,0,0,0],
#                   [0,np.pi,0,0],
#                   [0,np.pi,np.pi,0],
#                   [0,0,np.pi,0]])

# A_pi = np.ones((4,4))


# region Single momentum point Hamiltonian construction


def M_pi_mag_sub_AB_single(k, h, n, theta):
    zmag = contract('k,ik->i', n, z)
    ffact = contract('k, jk->j', k, NN)
    ffact = np.exp(1j * ffact)
    M = contract('u, u, ru, urx->rx', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                 np.exp(1j * A_pi), piunitcell)
    return M


def M_pi_sub_intrahopping_AA_single(k, alpha, eta, Jpm):
    ffact = contract('k, jlk->jl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    M = contract('jl,kjl,jl, jka, lkb->ab', notrace, -Jpm * A_pi_rs_traced / 4 * eta[alpha], ffact, piunitcell,
                 piunitcell)
    return M


def M_pi_sub_interhopping_AB_single(k, alpha, Jpmpm, xi):
    ffact = contract('k, jk->j', k, NN)
    ffact = np.exp(1j * neta(alpha) * ffact)
    tempxa = xi[alpha]
    tempxb = xi[1 - alpha]
    M1a = contract('jl, kjl, j, kl, jkx->kx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa, piunitcell)
    M1b = contract('jl, kjl, l, kj, lkx->kx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa, piunitcell)
    M2a = contract('jl, kjl, j, kl, jkx->xk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),
                   piunitcell)
    M2b = contract('jl, kjl, l, kj, lkx->xk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),
                   piunitcell)
    return M1a + M1b + M2a + M2b


def M_pi_sub_pairing_AA_single(k, alpha, Jpmpm, chi, chi0):
    di = np.identity(4)
    ffact = contract('k, jlk->jl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    beta = 1 - alpha
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl,km->km', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, di)
    M2 = contract('jl, kjl, jl, k, jka, lkb->ba', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0, piunitcell,
                  piunitcell)
    return M1 + M2


def M_pi_single(k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])

    dummy = np.zeros((4, 4))

    MAk = M_pi_sub_intrahopping_AA_single(k, 0, eta, Jpm)
    MBk = M_pi_sub_intrahopping_AA_single(k, 1, eta, Jpm)
    MAnk = M_pi_sub_intrahopping_AA_single(-k, 0, eta, Jpm)
    MBnk = M_pi_sub_intrahopping_AA_single(-k, 1, eta, Jpm)

    temp = M_pi_mag_sub_AB_single(k, h, n, theta)
    temp1 = M_pi_sub_interhopping_AB_single(k, 0, Jpmpm, xi)
    MagAkBk = temp + temp1
    MagBkAk = np.conj(np.transpose(MagAkBk))
    MagAnkBnk = M_pi_mag_sub_AB_single(-k, h, n, theta) + M_pi_sub_interhopping_AB_single(-k, 0, Jpmpm, xi)
    MagBnkAnk = np.conj(np.transpose(MagAnkBnk))

    MAdkAdnk = M_pi_sub_pairing_AA_single(k, 0, Jpmpm, chi, chi0)
    MBdkBdnk = M_pi_sub_pairing_AA_single(k, 1, Jpmpm, chi, chi0)
    MAnkAk = np.conj(np.transpose(MAdkAdnk))
    MBnkBk = np.conj(np.transpose(MBdkBdnk))

    FM = np.block([[MAk, MagAkBk, MAdkAdnk, dummy],
                   [MagBkAk, MBk, dummy, MBdkBdnk],
                   [MAnkAk, dummy, MAnk, MagAnkBnk],
                   [dummy, MBnkBk, MagBnkAnk, MBnk]])

    return FM


# endregion

#region Hamiltonian Construction
def M_pi_mag_sub_AB(k, h, n, theta):
    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    M = contract('ku, u, ru, urx->krx', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                 np.exp(1j*A_pi), piunitcell)
    return M


def M_pi_sub_intrahopping_AA(k, alpha, eta, Jpm):
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    M = contract('jl,kjl,ijl, jka, lkb->iab', notrace, -Jpm * A_pi_rs_traced / 4 * eta[alpha], ffact, piunitcell,
                 piunitcell)
    return M


def M_pi_sub_interhopping_AB(k, alpha, Jpmpm, xi):
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * neta(alpha) * ffact)
    tempxa = xi[alpha]
    tempxb = xi[1 - alpha]
    M1a = contract('jl, kjl, ij, kl, jkx->ikx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa, piunitcell)
    M1b = contract('jl, kjl, il, kj, lkx->ikx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa, piunitcell)
    M2a = contract('jl, kjl, ij, kl, jkx->ixk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),
                   piunitcell)
    M2b = contract('jl, kjl, il, kj, lkx->ixk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),
                   piunitcell)
    return M1a + M1b + M2a + M2b


def M_pi_sub_pairing_AA(k, alpha, Jpmpm, chi, chi0):
    d = np.ones(len(k))
    di = np.identity(4)
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    beta = 1 - alpha
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, i, km->ikm', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, d, di)
    M2 = contract('jl, kjl, ijl, k, jka, lkb->iba', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0, piunitcell,
                  piunitcell)
    return M1 + M2


def M_pi(k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])

    dummy = np.zeros((len(k), 4, 4))

    MAk = M_pi_sub_intrahopping_AA(k, 0, eta, Jpm)
    MBk = M_pi_sub_intrahopping_AA(k, 1, eta, Jpm)
    MAnk = M_pi_sub_intrahopping_AA(-k, 0, eta, Jpm)
    MBnk = M_pi_sub_intrahopping_AA(-k, 1, eta, Jpm)

    temp = M_pi_mag_sub_AB(k, h, n, theta)
    temp1 = M_pi_sub_interhopping_AB(k, 0, Jpmpm, xi)
    MagAkBk = temp + temp1
    MagBkAk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))
    MagAnkBnk = M_pi_mag_sub_AB(-k, h, n, theta) + M_pi_sub_interhopping_AB(-k, 0, Jpmpm, xi)
    MagBnkAnk = np.conj(np.transpose(MagAnkBnk, (0, 2, 1)))

    MAdkAdnk = M_pi_sub_pairing_AA(k, 0, Jpmpm, chi, chi0)
    MBdkBdnk = M_pi_sub_pairing_AA(k, 1, Jpmpm, chi, chi0)
    MAnkAk = np.conj(np.transpose(MAdkAdnk, (0, 2, 1)))
    MBnkBk = np.conj(np.transpose(MBdkBdnk, (0, 2, 1)))

    FM = np.block([[MAk, MagAkBk, MAdkAdnk, dummy],
                   [MagBkAk, MBk, dummy, MBdkBdnk],
                   [MAnkAk, dummy, MAnk, MagAnkBnk],
                   [dummy, MBnkBk, MagBnkAnk, MBnk]])

    return FM

#endregion

def E_pi_fixed(lams, M):
    M = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(M)
    return [E, V]


def E_pi(lams, k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    M = M_pi(k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
    M = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    # M = M_pi_alg(k,Jpm, Jpmpm, h, n, lams[0])
    E, V = np.linalg.eigh(M)
    return [E, V]


def rho_true(Jzz, M, lams):
    temp = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(temp)
    Vt = np.real(contract('ijk,ijk->ijk', V, np.conj(V)))
    Ep = contract('ijk, ik->ij', Vt, Jzz / np.sqrt(2 * Jzz * E))
    return np.mean(Ep) * np.ones(2)

def rho_true_site(Jzz, M, lams):
    # dumb = np.array([[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1]])
    # print(M)
    temp = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(temp)
    Vt = np.real(contract('ijk,ijk->ijk', V, np.conj(V)))
    Ep = contract('ijk, ik->ij', Vt, Jzz / np.sqrt(2 * Jzz * E))
    return np.mean(Ep, axis=0)[0:8]



def findminLam_old(M, Jzz, tol):
    warnings.filterwarnings("error")
    lamMin = np.zeros(2)
    lamMax = 50 * np.ones(2)
    lams = (lamMin + lamMax) / 2
    while not ((lamMax - lamMin <= tol).all()):
        lams = (lamMin + lamMax) / 2
        try:
            rhoguess = rho_true(Jzz, M, lams)
            for i in range(2):
                lamMax[i] = lams[i]
        except:
            lamMin = lams
    warnings.resetwarnings()
    return lams


def E_pi_single(lams, k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    M = M_pi_single(k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
    M = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(M)
    return [E, V]


def Emin(k, lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    return E_pi_single(lams, k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0][0]


def gradient(k, lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    kx, ky, kz = k
    step = 1e-8
    fx = (Emin(np.array([kx + step, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi) - Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)) / step
    fy = (Emin(np.array([kx, ky + step, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi) - Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)) / step
    fz = (Emin(np.array([kx, ky, kz + step]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi) - Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)) / step
    return np.array([fx, fy, fz])


def findminLam(M, K, tol, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    warnings.filterwarnings("error")
    E, V = np.linalg.eigh(M)
    E = np.around(E[:,0], decimals=14)
    Em = E.min()
    dex = np.where(E == Em)
    Know = np.unique(K[dex], axis=0)
    # print(Know)
    if Know.shape == (3,):
        Know = Know.reshape(1,3)

    if (E==0).all():
        return 0, Know
    step = 1


    for i in range(len(Know)):
        stuff = True
        init = True
        Enow = Em
        while stuff:
            if not init:
                gradlen = gradient(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi) - gradient(Klast,
                                                                                                                 np.zeros(
                                                                                                                     2),
                                                                                                                 eta,
                                                                                                                 Jpm,
                                                                                                                 Jpmpm,
                                                                                                                 h, n,
                                                                                                                 theta,
                                                                                                                 chi,
                                                                                                                 chi0,
                                                                                                                 xi)
                try:
                    step = abs(np.dot(Know[i] - Klast, gradlen)) / np.linalg.norm(gradlen) ** 2
                except:
                    step = 0

            Klast = np.copy(Know[i])
            Know[i] = Know[i] - step * gradient(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
            Elast = Enow
            Enow = Emin(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
            init = False
            if (abs(Enow - Elast) < 1e-12):
                stuff = False
    warnings.resetwarnings()
    return -Enow, Know


def findlambda_pi(M, Jzz, kappa, tol):
    warnings.filterwarnings("error")
    lamMin = np.zeros(2)
    lamMax = 50 * np.ones(2)
    lams = (lamMin + lamMax) / 2
    rhoguess = rho_true(Jzz, M, lams)
    # print(self.kappa)

    while not ((np.absolute(rhoguess - kappa) <= tol).all()):
        # for i in range(2):
        lams = (lamMin + lamMax) / 2
        # rhoguess = rho_true(Jzz, M, lams)
        try:
            rhoguess = rho_true(Jzz, M, lams)
            # rhoguess = self.rho_zero(alpha, self.lams)
            for i in range(2):
                if rhoguess[i] - kappa > 0:
                    lamMin[i] = lams[i]
                else:
                    lamMax[i] = lams[i]
        except:
            lamMin = lams

    warnings.resetwarnings()
    return lams


# region calculate mean field

# xipicell_0 = np.array([[1,-1,0,0],
#                        [-1,-1,0,0],
#                        [0,0,-1,1],
#                        [0,0,1,1]])


def chiCal(lams, M, K, Jzz):
    E, V = E_pi_fixed(lams, M)
    E = np.sqrt(2 * Jzz * E)
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jlk->ijl', K, NNminus)
    ffactB = np.exp(-1j * ffact)
    A = contract('iab, ijl,jka, lkb->ikjl', green[:, 8:12, 0:4], ffactB, piunitcell, piunitcell)
    B = np.where(A!=0)
    M1 = np.mean(A, axis=0)
    chi = M1[0, 0, 3]
    chi0 = np.conj(M1[0, 0, 0])
    return chi, chi0


# def chi0(lams, M, Jzz):
#     E, V = E_pi_fixed(lams, M)
#     E = np.sqrt(2*Jzz*E)
#     green = green_pi(E, V, Jzz)
#
#     chi0A = np.mean(green[:, 0, 8]) * np.ones(4)
#     # chi0B = np.mean(green[:, 4, 12]) * np.ones(4)
#
#     return np.array([chi0A, chi0A])

def xiCal(lams, M, K, Jzz, ns):
    E, V = E_pi_fixed(lams, M)
    E = np.sqrt(2 * Jzz * E)
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jk->ij', K, NN)
    ffactA = np.exp(1j * ffact)

    M1 = np.mean(contract('ika, ij,jka->ikj', green[:, 0:4, 4:8], ffactA, piunitcell), axis=0)

    M1 = M1[0, 0]
    return np.real(M1)


# endregion


# graphing BZ

def dispersion_pi(lams, k, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi):
    temp = np.sqrt(2 * Jzz * E_pi(lams, k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])
    return temp


def calDispersion(lams, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi):
    dGammaX = dispersion_pi(lams, GammaX, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi)
    dXW = dispersion_pi(lams, XW, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi)
    dWK = dispersion_pi(lams, WK, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi)
    dKGamma = dispersion_pi(lams, KGamma, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi)
    dGammaL = dispersion_pi(lams, GammaL, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi)
    dLU = dispersion_pi(lams, LU, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi)
    dUW = dispersion_pi(lams, UW, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi)

    for i in range(dGammaX.shape[1]):
        plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX[:, i], 'b')
        plt.plot(np.linspace(gX, gW1, len(dXW)), dXW[:, i], 'b')
        plt.plot(np.linspace(gW1, gK, len(dWK)), dWK[:, i], 'b')
        plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma[:, i], 'b')
        plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL[:, i], 'b')
        plt.plot(np.linspace(gL, gU, len(dLU)), dLU[:, i], 'b')
        plt.plot(np.linspace(gU, gW2, len(dUW)), dUW[:, i], 'b')

    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW1, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gK, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma2, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gL, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gU, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW2, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [gGamma1, gX, gW1, gK, gGamma2, gL, gU, gW2]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)


# @nb.njit(parallel=True, cache=True)
def minCal(lams, q, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi):
    temp = np.zeros(len(q))
    mins = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])[:, 0]
    for i in range(len(q)):
        temp[i] = np.min(
            np.sqrt(2 * Jzz * E_pi(lams, K - q[i], eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])[:, 0] + mins)
    return temp


# @nb.njit(parallel=True, cache=True)
def maxCal(lams, q, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi):
    temp = np.zeros(len(q))
    maxs = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])[:, -1]
    for i in range(len(q)):
        temp[i] = np.max(
            np.sqrt(2 * Jzz * E_pi(lams, K - q[i], eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])[:, -1] + maxs)
    return temp


def minMaxCal(lams, q, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi):
    temp = np.zeros((len(q), 2))
    Ek = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])
    for i in range(len(q)):
        Eq = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])
        temp[i, 0] = np.min(np.amin(Ek, axis=1) + np.amin(Eq, axis=1))
        temp[i, 1] = np.max(np.amax(Ek, axis=1) + np.amax(Eq, axis=1))
    return temp


def loweredge(lams, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi):
    dGammaX = minCal(lams, GammaX, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dXW = minCal(lams, XW, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dWK = minCal(lams, WK, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dLU = minCal(lams, LU, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dUW = minCal(lams, UW, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)

    plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, 'b')
    plt.plot(np.linspace(gX, gW1, len(dXW)), dXW, 'b')
    plt.plot(np.linspace(gW1, gK, len(dWK)), dWK, 'b')
    plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, 'b')
    plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, 'b')
    plt.plot(np.linspace(gL, gU, len(dLU)), dLU, 'b')
    plt.plot(np.linspace(gU, gW2, len(dUW)), dUW, 'b')

    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW2, color='w', label='axvline - full height', linestyle='dashed')
    xlabpos = [gGamma1, gX, gW1, gK, gGamma2, gL, gU, gW2]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)


def upperedge(lams, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi):
    dGammaX = maxCal(lams, GammaX, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dXW = maxCal(lams, XW, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dWK = maxCal(lams, WK, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dKGamma = maxCal(lams, KGamma, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dGammaL = maxCal(lams, GammaL, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dLU = maxCal(lams, LU, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)
    dUW = maxCal(lams, UW, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi)

    plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, 'b')
    plt.plot(np.linspace(gX, gW1, len(dXW)), dXW, 'b')
    plt.plot(np.linspace(gW1, gK, len(dWK)), dWK, 'b')
    plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, 'b')
    plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, 'b')
    plt.plot(np.linspace(gL, gU, len(dLU)), dLU, 'b')
    plt.plot(np.linspace(gU, gW2, len(dUW)), dUW, 'b')

    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW2, color='w', label='axvline - full height', linestyle='dashed')
    xlabpos = [gGamma1, gX, gW1, gK, gGamma2, gL, gU, gW2]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)


def gap(M, lams):
    temp = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(temp)
    # E = np.sqrt(E)
    temp = np.amin(E)
    return temp


def EMAX(M, lams):
    temp = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(temp)
    temp = np.amax(E)
    return temp


def green_pi_phid_phi(E, V, Jzz):
    Vt1 = contract('ijk, ilk->iklj', V[:, :, 0:8], np.conj(V)[:, :, 0:8])
    Vt2 = contract('ijk, ilk->iklj', V[:, :, 8:16], np.conj(V)[:, :, 8:16])
    green = Jzz / E
    green1 = contract('ikjl, ik->ijl', Vt1, green[:, 0:8])
    green2 = contract('iklj, ik->ijl', Vt2, green[:, 8:16])

    return green1 + green2


def green_pi_phid_phid(E, V, Jzz):
    V = np.conj(np.transpose(V, (0, 2, 1)))
    Vt1 = contract('ijk, ilk->ikjl', V[:, 0:8, 0:8], V[:, 0:8, 8:16])
    Vt2 = contract('ijk, ilk->ikjl', V[:, 0:8, 8:16], V[:, 0:8, 0:8])
    green = Jzz / E
    green1 = contract('ikjl, ik->ijl', Vt1, green[:, 0:8])
    green2 = contract('iklj, ik->ijl', Vt2, green[:, 8:16])
    return green1 + green2


def green_pi_wrong(E, V, Jzz):
    green_phid_phid = green_pi_phid_phid(E, V, Jzz)
    green_phi_phi = np.transpose(np.conj(green_phid_phid), (0, 2, 1))
    green_phid_phi = green_pi_phid_phi(E, V, Jzz)
    green = np.block([[green_phid_phi[:, 0:8, 0:8], green_phid_phid],
                      [green_phi_phi, green_phid_phi[:, 8:16, 8:16]]])

    return green


def green_pi(E, V, Jzz):
    Vt = contract('ijk, ilk->iklj', V, np.conj(V))
    green = Jzz / E
    green = contract('ikjl, ik->ijl', Vt, green)
    return green


# region outdated
def green_pi_phid_phi_branch(E, V, Jzz):
    Vt1 = contract('ijk, ikl->iklj', V[:, :, 0:8], np.transpose(np.conj(V), (0, 2, 1))[:, 0:8, :])
    Vt2 = contract('ijk, ikl->iklj', V[:, :, 8:16], np.transpose(np.conj(V), (0, 2, 1))[:, 8:16, :])
    green = Jzz / E
    green1 = contract('ikjl, ik->ikjl', Vt1, green[:, 0:8])
    green2 = contract('iklj, ik->ikjl', Vt2, green[:, 8:16])
    green = np.zeros((len(E), 16, 16, 16))
    green[:, 0:8] = green1
    green[:, 8:16] = green2
    return green


def green_pi_phi_phi_branch(E, V, Jzz):
    Vt1 = contract('ijk, ilk->ikjl', V[:, 0:8, 8:16], V[:, 0:8, 0:8])
    Vt2 = contract('ijk, ilk->ikjl', V[:, 0:8, 0:8], V[:, 0:8, 8:16])
    green = Jzz / E
    green1 = contract('ikjl, ik->ikjl', Vt1, green[:, 8:16])
    green2 = contract('iklj, ik->ikjl', Vt2, green[:, 0:8])
    green = np.zeros((len(E), 16, 8, 8))
    green[:, 0:8] = green2
    green[:, 8:16] = green1
    return green


def green_pi_branch_wrong(E, V, Jzz):
    green_phi_phi = green_pi_phi_phi_branch(E, V, Jzz)
    green_phid_phid = np.transpose(np.conj(green_phi_phi), (0, 1, 3, 2))
    green_phid_phi = green_pi_phid_phi_branch(E, V, Jzz)
    green = np.block([[green_phid_phi[:, :, 0:8, 0:8], green_phid_phid],
                      [green_phi_phi, green_phid_phi[:, :, 8:16, 8:16]]])
    return green


# endregion

def green_pi_branch(E, V, Jzz):
    Vt = contract('ijk, ilk->iklj', V, np.conj(V))
    green = Jzz / E
    green = contract('ikjl, ik->ikjl', Vt, green)
    return green


def MFE(Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, M, lams, k):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])

    M = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(M)
    E = np.sqrt(2 * Jzz * E)
    Vt = contract('ijk, ilk->iklj', V, np.conj(V))
    green = green_pi(E, V, Jzz)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffactA = np.exp(-1j * ffact)
    ffactB = np.exp(1j * ffact)

    EQ = np.real(np.trace(np.mean(contract('ikjl, ik->ijl', Vt, E / 2), axis=0)) / 2)

    E1A = np.mean(
        contract('jl,kjl, iab, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced / 4, green[:, 0:4, 0:4], ffactA,
                 piunitcell, piunitcell), axis=0)
    E1B = np.mean(
        contract('jl,kjl, iab, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced / 4, green[:, 4:8, 4:8], ffactB,
                 piunitcell, piunitcell), axis=0)

    # print(E1A)
    E1 = np.real(E1A + E1B)

    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    Emag = np.mean(contract('ku, u, ru, krx, urx->k', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                            np.exp(1j * A_pi), green[:, 0:4, 4:8], piunitcell), axis=0)

    Emag = 2 * np.real(Emag)

    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    tempxb = xi[1]
    tempxa = xi[0]
    M1a = np.mean(contract('jl, kjl, ij, kl, ikx, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa,
                           green[:, 0:4, 4:8], piunitcell), axis=0)
    M1b = np.mean(contract('jl, kjl, il, kj, ikx, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa,
                           green[:, 0:4, 4:8], piunitcell), axis=0)
    M2a = np.mean(
        contract('jl, kjl, ij, kl, ixk, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),
                 green[:, 0:4, 4:8], piunitcell), axis=0)
    M2b = np.mean(
        contract('jl, kjl, il, kj, ixk, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),
                 green[:, 0:4, 4:8], piunitcell), axis=0)
    EAB = 2 * np.real(M1a + M1b + M2a + M2b)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    beta = 1
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = np.mean(contract('jl, kjl, kjl, ikk->i', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, green[:, 0:4, 8:12]),
                 axis=0)

    M2 = np.mean(contract('jl, kjl, ijl, k, iba, jka, lkb->i', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0,
                          green[:, 0:4, 8:12], piunitcell,
                          piunitcell), axis=0)

    EAA = 2 * np.real(M1 + M2)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * ffact)
    beta = 0
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = np.mean(
        contract('jl, kjl, kjl, ikk->i', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, green[:, 4:8, 12:16]), axis=0)

    M2 = np.mean(contract('jl, kjl, ijl, k, iba, jka, lkb->i', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0,
                          green[:, 4:8, 12:16], piunitcell,
                          piunitcell), axis=0)

    EBB = 2 * np.real(M1 + M2)

    E = EQ + Emag + E1 + EAB + EAA + EBB
    # print(EQ/4, E1/4, Emag/4, EAB/4, EAA/4, EBB/4)
    return E / 4

def MFE_condensed(Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, M, lams, k, rhos):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])

    ffact = contract('k, jlk->jl', k, NNminus)
    ffactA = np.exp(-1j * ffact)
    ffactB = np.exp(1j * ffact)


    E1A = contract('jl,kjl, a, b, jl, jka, lkb->', notrace, -Jpm * A_pi_rs_traced / 4, rhos[0:4], rhos[0:4], ffactA,
                 piunitcell, piunitcell)
    E1B = contract('jl,kjl, a, b, jl, jka, lkb->', notrace, -Jpm * A_pi_rs_traced / 4, rhos[4:8], rhos[4:8], ffactB,
                 piunitcell, piunitcell)

    # print(E1A)
    E1 = np.real(E1A + E1B)

    zmag = contract('k,ik->i', n, z)
    ffact = contract('k, jk->j', k, NN)
    ffact = np.exp(1j * ffact)
    Emag = contract('u, u, ru, r, x, urx->', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                            np.exp(1j * A_pi), rhos[0:4], rhos[4:8], piunitcell)

    Emag = 2 * np.real(Emag)

    ffact = contract('k, jk->j', k, NN)
    ffact = np.exp(1j * ffact)
    tempxb = xi[1]
    tempxa = xi[0]
    M1a = contract('jl, kjl, j, kl, k, x, jkx->', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa,rhos[0:4], rhos[4:8], piunitcell)
    M1b = contract('jl, kjl, l, kj, k, x, lkx->', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa,rhos[0:4], rhos[4:8], piunitcell)
    M2a = contract('jl, kjl, j, kl, x, k, jkx->', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),rhos[0:4], rhos[4:8], piunitcell)
    M2b = contract('jl, kjl, l, kj, x, k, lkx->', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),rhos[0:4], rhos[4:8], piunitcell)
    EAB = 2 * np.real(M1a + M1b + M2a + M2b)

    ffact = contract('k, jlk->jl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    beta = 1
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, k, k->', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, rhos[0:4], rhos[0:4])
    M2 = contract('jl, kjl, jl, k, b, a, jka, lkb->', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0, rhos[0:4], rhos[0:4], piunitcell,
                          piunitcell)

    EAA = 2 * np.real(M1 + M2)

    ffact = contract('k, jlk->jl', k, NNminus)
    ffact = np.exp(1j * ffact)
    beta = 0
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, k, k->', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, rhos[4:8], rhos[4:8])
    M2 = contract('jl, kjl, jl, k, b, a, jka, lkb->', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0, rhos[4:8], rhos[4:8], piunitcell,
                          piunitcell)

    EBB = 2 * np.real(M1 + M2)

    E = Emag + E1 + EAB + EAA + EBB
    # print(EQ/4, E1/4, Emag/4, EAB, EAA, EBB)
    return E / 4


def MFE_condensed_0(Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, M, lams, k, rhos):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffactA = np.exp(-1j * ffact)
    ffactB = np.exp(1j * ffact)


    E1A = contract('jl,kjl, a, b, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced / 4, rhos[0:4], rhos[0:4], ffactA,
                 piunitcell, piunitcell)
    E1B = contract('jl,kjl, a, b, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced / 4, rhos[4:8], rhos[4:8], ffactB,
                 piunitcell, piunitcell)

    # print(E1A)
    E1 = np.real(E1A + E1B)

    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    Emag = contract('iu, u, ru, r, x, urx->i', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                            np.exp(1j * A_pi), rhos[0:4], rhos[4:8], piunitcell)

    Emag = 2 * np.real(Emag)

    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    tempxb = xi[1]
    tempxa = xi[0]
    M1a = contract('jl, kjl, ij, kl, k, x, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa,rhos[0:4], rhos[4:8], piunitcell)
    M1b = contract('jl, kjl, il, kj, k, x, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa,rhos[0:4], rhos[4:8], piunitcell)
    M2a = contract('jl, kjl, ij, kl, x, k, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),rhos[0:4], rhos[4:8], piunitcell)
    M2b = contract('jl, kjl, il, kj, x, k, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),rhos[0:4], rhos[4:8], piunitcell)
    EAB = 2 * np.real(M1a + M1b + M2a + M2b)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    beta = 1
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, k, k->', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, rhos[0:4], rhos[0:4])
    M2 = contract('jl, kjl, ijl, k, b, a, jka, lkb->', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0, rhos[0:4], rhos[0:4], piunitcell,
                          piunitcell)

    EAA = 2 * np.real(M1 + M2)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * ffact)
    beta = 0
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, k, k->', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, rhos[4:8], rhos[4:8])
    M2 = contract('jl, kjl, ijl, k, b, a, jka, lkb->', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0, rhos[4:8], rhos[4:8], piunitcell,
                          piunitcell)

    EBB = 2 * np.real(M1 + M2)

    E = np.mean(Emag + E1 + EAB + EAA + EBB)
    # print(EQ/4, E1/4, Emag/4, EAB, EAA, EBB)
    return E / 4

#
# def GS(lams, k, Jzz, Jpm, eta, h, n):
#     return np.mean(dispersion_pi(lams, k, Jzz, Jpm, eta), axis=0) - np.repeat(lams)

class piFluxSolver:

    def __init__(self, Jxx, Jyy, Jzz, theta=0, h=0, n=np.array([0, 0, 0]), eta=1, kappa=2, lam=2, BZres=20, graphres=20,
                 ns=1):
        self.Jzz = Jzz
        self.Jpm = -(Jxx + Jyy) / 4
        self.Jpmpm = (Jxx - Jyy) / 4
        self.theta = theta
        self.kappa = kappa
        self.eta = np.array([eta, 1], dtype=float)
        self.tol = 1e-5
        self.lams = np.array([lam, lam], dtype=np.double)
        self.ns = ns
        self.h = h
        self.n = n
        self.chi = 1
        # self.chi0 = np.zeros(8, dtype=np.complex128)
        self.xi = 1
        self.chi0 = 1

        self.minLams = np.zeros(2, dtype=np.double)

        self.BZres = BZres
        self.graphres = graphres
        self.bigB = np.concatenate((genBZ(BZres), symK))
        self.bigB = np.unique(self.bigB, axis=0)
        # self.bigB = genBZ(BZres)
        self.bigTemp = np.copy(self.bigB)
        self.MF = M_pi(self.bigB, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
                       self.xi)
        self.MForig = np.copy(self.MF)
        self.q = np.nan
        self.qmin = np.empty(3)
        self.qmin[:] = np.nan
        self.condensed = False

        self.delta = np.zeros(2)
        self.rhos = np.zeros(4)

    # alpha = 1 for A = -1 for B

    def findLambda(self):
        self.lams = findlambda_pi(self.MF, self.Jzz, self.kappa, self.tol)

    def findminLam_old(self):
        return findminLam_old(self.MF, self.Jzz, 1e-10)

    def findminLam(self):
        minLams, self.qmin = findminLam(self.MF, self.bigB, self.tol, self.eta, self.Jpm, self.Jpmpm, self.h, self.n,
                                        self.theta, self.chi, self.chi0, self.xi)
        self.minLams = np.ones(2) * minLams
        self.bigTemp = np.unique(np.concatenate((self.bigB, self.qmin)), axis=0)
        self.MF = M_pi(self.bigTemp, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
                       self.xi)
    def calmeanfield(self):
        # cond = self.ifcondense(self.bigTemp)
        # # leng = len(self.bigTemp)
        # Kps = np.delete(self.bigTemp, cond, axis=0)
        # MFp = np.delete(self.MF, cond, axis=0)
        # if self.condensed:
        #     MFq = M_pi(self.qmin, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
        #                self.xi)
        #     chic, chi0c = np.array(chiCal(self.delta, MFq, self.qmin, self.Jzz))
        #     xic = xiCal(self.delta, MFq, self.qmin, self.Jzz, self.ns)
        #     chi, chi0 = chiCal(self.minLams, MFp, Kps, self.Jzz)
        #     xi = xiCal(self.minLams, MFp, Kps, self.Jzz, self.ns)
        #     return chi + chic, chi0 + chi0c, xi + xic
        # else:
        MFp = self.MF
        Kps = self.bigTemp
        chi, chi0 = chiCal(self.lams, MFp, Kps, self.Jzz)
        xi = xiCal(self.lams, MFp, Kps, self.Jzz, self.ns)
        return chi, chi0, xi

    def solvemeanfield(self, tol=1e-7):
        self.condensation_check()
        chinext, chi0next, xinext = self.calmeanfield()
        # J0 = self.Jacobian(np.array([chinext, chi0next, xinext]))

        while ((abs(chinext - self.chi) >= tol).any() or (abs(xinext - self.xi) >= tol).any() or (
                abs(chi0next - self.chi0) >= tol).any()):
            # print(self.lams, self.chi, self.chi0, self.xi, self.MFE())

            # fn = self.SCE(chinext, chi0next, xinext)
            # dF = fn - self.SCE(self.chi, self.chi0, self.xi)
            # dX = np.array([chinext-self.chi, chi0next-self.chi0, xinext-self.xi])
            #
            # J0 = J0 + contract('i,j->ij',dF-np.matmul(J0, dX), dX)/np.linalg.norm(dX)**2
            # print(J0)
            self.chi = chinext
            self.chi0 = chi0next
            self.xi = xinext
            self.MF = M_pi(self.bigB, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
                           self.xi)
            self.condensation_check()

            # if (J0<1e-5).any():
            chinext, chi0next, xinext = self.calmeanfield()
            # else:
            # chinext, chi0next, xinext = np.array([self.chi, self.chi0, self.xi]) - np.matmul(np.linalg.inv(J0), fn)
            print(self.chi, self.chi0, self.xi)
        self.chi = chinext
        self.chi0 = chi0next
        self.xi = xinext
        return 0

    def qvec(self):
        E = \
        E_pi(np.zeros(2), self.bigB, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
             self.xi)[0][:, 0]
        A = np.where(E == E.min())
        self.q = self.bigB[A]

    def ifcondense(self, q, tol=0):
        c = np.array([])
        if self.condensed:
            E, V = self.LV_zero(q,self.minLams)
            E = E[:,0]
            c = np.where(E<=tol)[0]
        c = np.array(c, dtype=int)
        return c

    def low(self):
        E, V = np.linalg.eigh(self.MF)
        cond = np.argmin(E[:, 0])
        return self.bigB[cond], E[cond][0]

    def set_condensed(self):
        A = -self.minLams[0] + self.lams[0]
        # B = (2e2 / len(self.bigB)) ** 2
        self.condensed = A < (1e2 / len(self.bigTemp)) ** 2

    def condensation_check(self):
        self.findminLam()
        self.findLambda()
        self.set_condensed()
        self.set_delta()

    def M_true(self, k):
        return M_pi(k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)

    def E_pi(self, k):
        return np.sqrt(2 * self.Jzz *
                       E_pi(self.lams, k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi,
                            self.chi0, self.xi)[0])

    def Eigen_single_lam0(self, k):
        return \
        E_pi_single(np.zeros(2), k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
                    self.xi)[0]

    def E_pi_single(self, k):
        return np.sqrt(2 * self.Jzz *
                       E_pi_single(self.lams, k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi,
                                   self.chi0, self.xi)[0])

    def E_pi_fixed(self):
        return np.sqrt(2 * self.Jzz * E_pi_fixed(self.lams, self.MF)[0])

    def dispersion(self, k):
        return dispersion_pi(self.lams, k, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta,
                             self.chi, self.chi0, self.xi)

    def LV_zero(self, k, lam=np.zeros(2)):
        if np.any(lam == 0):
            lam = self.lams
        return E_pi(lam, k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi)

    def gap(self):
        return np.sqrt(2 * self.Jzz * gap(self.MF, self.lams))

    def gapwhere(self):
        temp = self.MF + np.diag(np.repeat(self.lams, 4))
        E, V = np.linalg.eigh(temp)
        # E = np.sqrt(2*self.Jzz*E)
        dex = np.argmin(E[:, 0])
        return np.mod(self.bigB[dex], 2 * np.pi)


    def mag_con(self):
        return np.mean(E_pi_fixed(np.zeros(2), self.MF)[0])

    def GS(self):
        return np.mean(self.E_pi(self.bigB)) - np.mean(self.lams)

    def set_delta(self):
        if self.condensed:
            self.delta = (self.lams-self.minLams)*len(self.bigTemp)**2
            cond = self.ifcondense(self.bigTemp, self.gap()**2/(2*self.Jzz))
            MFp = np.delete(self.MF, cond, axis=0)
            self.rhos = np.sqrt(self.kappa - rho_true_site(self.Jzz, MFp,self.minLams))

            # print(self.rhos)

    def MFE(self, chi=-10, chi0=-10, xi=-10):
        if chi == -10:
            chi = self.chi
        if chi0 == -10:
            chi0 = self.chi0
        if xi == -10:
            xi = self.xi
        # if self.condensed:
        #     Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi,
        #              self.MF,
        #              self.lams, self.bigTemp)
        #
        #     a = np.mod(self.qmin[0], 2*np.pi)
        #     for i in range(3):
        #         if abs(abs(a[i]) - 2*np.pi) < 5e-6:
        #             a[i] = 0
        #
        #     Eq = MFE_condensed(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
        #                        self.xi, self.MF,
        #                        self.minLams, a, self.rhos)
        #     return Ep + Eq
        # else:
        Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.MF,
        self.lams, self.bigTemp)
        return Ep

    def SCE(self, chi, chi0, xi):
        tol = 1e-3
        temp = self.MFE(chi, chi0, xi)
        tempChi = self.MFE(chi + tol, chi0, xi)
        tempChi0 = self.MFE(chi, chi0 + tol, xi)
        tempXi = self.MFE(chi, chi0, xi + tol)
        return (np.array([tempChi, tempChi0, tempXi]) - temp) / tol

    def Jacobian(self, mfp):
        mfps = mfp + 1e-3
        chi, chi0, xi = mfp
        chil, chi0l, xil = mfps
        temp = self.SCE(chil, chi0l, xil)
        tempChi = self.SCE(chi, chi0l, xil)
        tempChi0 = self.SCE(chil, chi0, xil)
        tempXi = self.SCE(chil, chi0l, xi)
        stuff = np.array([tempChi, tempChi0, tempXi]) - temp
        dX = 1 / (mfp - mfps)
        J = contract('ux,u->ux', stuff, dX)
        return J

    def graph(self, show):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta, self.chi,
                      self.chi0, self.xi)
        if show:
            plt.show()

    def minCal(self, K):
        return minCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.bigB, self.theta,
                      self.chi, self.chi0, self.xi)

    def maxCal(self, K):
        return maxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.bigB, self.theta,
                      self.chi, self.chi0, self.xi)

    def minMaxCal(self, K):
        return minMaxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.bigB, self.theta,
                         self.chi, self.chi0, self.xi)

    def EMAX(self):
        return np.sqrt(2 * self.Jzz * EMAX(self.MF, self.lams))

    def TWOSPINON_GAP(self, k):
        return np.min(
            minCal(self.lams, k, self.Jzz, self.Jpmpm, self.Jpm, self.eta, self.h, self.n, self.bigB, self.theta,
                   self.chi, self.chi0, self.xi))

    def TWOSPINON_MAX(self, k):
        return np.max(
            maxCal(self.lams, k, self.Jzz, self.Jpmpm, self.Jpm, self.eta, self.h, self.n, self.bigB, self.theta,
                   self.chi, self.chi0, self.xi))

    def graph_loweredge(self, show):
        loweredge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.bigB, self.theta, self.chi,
                  self.chi0, self.xi)
        if show:
            plt.show()

    def graph_upperedge(self, show):
        upperedge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.bigB, self.theta, self.chi,
                  self.chi0, self.xi)
        if show:
            plt.show()

    def green_pi(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi(E, V, self.Jzz)

    # def green_pi_old(self, k, lam=np.zeros(2)):
    #     E, V = self.LV_zero(k, lam)
    #     E = np.sqrt(2 * self.Jzz * E)
    #     return green_pi_old(E, V, self.Jzz)

    def green_pi_branch(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi_branch(E, V, self.Jzz), E

    def magnetization(self):
        green = self.green_pi(self.bigTemp)
        ffact = contract('ik, jk->ij', self.bigTemp, NN)
        ffactp = np.exp(1j * ffact)
        ffactm = np.exp(-1j * ffact)

        magp = contract('ij, ika, kj, jka->i', ffactp, green[:, 0:4, 4:8], np.exp(1j * A_pi),
                        piunitcell) / 4
        magm = contract('ij, iak, kj, jka->i', ffactm, green[:, 4:8, 0:4], np.exp(-1j * A_pi),
                        piunitcell) / 4

        con = 0
        if self.condensed:

            a = np.mod(self.qmin[0], 2*np.pi)
            for i in range(3):
                if abs(abs(a[i]) - 2*np.pi) < 5e-6:
                    a[i] = 0

            ffact = contract('k, jk->j', a)
            ffactp = np.exp(1j * ffact)
            ffactm = np.exp(-1j * ffact)

            magp = contract('j, k, a, kj, jka->j', ffactp, self.rhos[0:4], self.rhos[4:8], np.exp(1j * A_pi),
                            piunitcell) / 4
            magm = contract('j, a, k, kj, jka->j', ffactm, self.rhos[4:8], self.rhos[0:4], np.exp(-1j * A_pi),
                            piunitcell) / 4

            con = np.mean(magp+magm)

        return np.real(np.mean(magp + magm)+con) / 4