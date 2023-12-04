import time

import matplotlib.pyplot as plt
import warnings

import numpy as np

from misc_helper import *



#region Hamiltonian Construction
def M_pi_mag_sub_AB(k, h, n, theta):
    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(-1j * ffact)
    M = contract('ku, u, ru, urx->krx', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                 np.exp(1j*A_pi), piunitcell)
    return M


def M_pi_sub_intrahopping_AA(k, alpha, eta, Jpm):
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    M = contract('jl,klj,ijl, jka, lkb->iab', notrace, -Jpm * A_pi_rs_traced / 4, ffact, piunitcell,
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



def rho_true_zeroed(lams, Jzz, M, kappa):
    #assumes same lambda for all sites and sublattices
    temp = M + np.diag(np.repeat(lams, 16))
    E, V = np.linalg.eigh(temp)
    Vt = np.real(contract('ijk,ijk->ijk', V, np.conj(V)))
    Ep = contract('ijk, ik->ij', Vt, Jzz / np.sqrt(2 * Jzz * E))
    a = np.mean(Ep) - kappa
    return a

def frho(lams, Jzz, M, kappa):
    tol=1e-16
    return (rho_true_zeroed(lams+tol, Jzz, M, kappa) - rho_true_zeroed(lams, Jzz, M, kappa))/tol

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


def Emin(q, lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    k = np.array(q).reshape((1,3))
    return E_pi(lams, k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0][0,0]


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
    if Jpm==0 and Jpmpm == 0 and h == 0:
        return 0, np.array([0,0,0]).reshape((1,3))
    warnings.filterwarnings("error")
    E, V = np.linalg.eigh(M)
    E = np.around(E[:,0], decimals=15)
    Em = E.min()
    dex = np.where(E==Em)
    Know = K[dex]

    if Know.shape == (3,):
        Know = Know.reshape(1,3)

    if len(Know) >= number:
        Know = Know[0:number]

    step = 1
    Enow = Em*np.ones(len(Know))


    for i in range(len(Know)):
        stuff = True
        init = True
        while stuff:
            # print(Enow[i], Know[i], i, len(Know))
            if not init:
                gradlen = gradient(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi) - gradient(Klast,
                        np.zeros(2),eta,Jpm,Jpmpm, h, n,theta,chi,chi0,xi)
                try:
                    step = abs(np.dot(Know[i] - Klast, gradlen)) / np.linalg.norm(gradlen) ** 2
                except:
                    step = 0

            Klast = np.copy(Know[i])
            Know[i] = Know[i] - step * gradient(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
            Elast = np.copy(Enow[i])
            Enow[i] = Emin(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
            init = False
            # print(Enow[i], Know[i])
            if abs(Elast-Enow[i])<1e-12:
                stuff = False
    warnings.resetwarnings()
    a = np.argmin(Enow)
    Know = np.mod(Know[a], 2*np.pi).reshape((1,3))
    for i in range(3):
        if (Know[0,i] > np.pi):
            Know[0,i] = Know[0,i] - 2*np.pi
    # Know = contract('i, ik->k', Know, BasisBZA).reshape((1,3))
    # print(Know, -Enow[a])
    return -Enow[a], Know

def findminLam_scipy(M, K, tol, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    if Jpm==0 and Jpmpm == 0 and h == 0:
        return 0, np.array([0,0,0]).reshape((1,3))

    E, V = np.linalg.eigh(M)
    E = np.around(E[:,0], decimals=15)
    Em = E.min()
    dex = np.where(E==Em)
    Know = K[dex]


    if Know.shape == (3,):
        Know = Know.reshape(1,3)

    if len(Know) >= number:
        Know = Know[0:number]

    Enow = np.zeros(len(Know))

    for i in range(len(Know)):
        res = minimize(Emin, x0=Know[i], args=(np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi), method='Nelder-Mead')
        Know[i] = np.array(res.x)
        Enow[i] = res.fun

    a = np.argmin(Enow)
    Know = np.mod(Know[a], 2*np.pi).reshape((1,3))
    for i in range(3):
        if Know[0,i] > np.pi:
            Know[0,i] = Know[0,i] - 2*np.pi
    return -Enow[a], Know

def findminLam_adam(M, K, tol, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    if Jpm==0 and Jpmpm == 0 and h == 0:
        return 0, np.array([0,0,0]).reshape((1,3))

    Know = np.pi*np.array([1, 1, 1])
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.001
    eps = 1e-8

    stuff = True
    m = 0
    v = 0
    t = 0

    while stuff:
        print(Know)
        t = t + 1
        g = gradient(Know, np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
        m = beta1 * m + (1-beta1) * g
        v = beta2 * v + (1-beta2) * g**2
        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)
        Klast = np.copy(Know)
        Know = Know - alpha * mhat / (np.sqrt(vhat) + eps)
        if (abs(Klast-Know)<1e-8).all():
            break

    Enow = Emin(Know, np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
    return -Enow, Know

def findminLam_momentum(M, K, tol, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi):
    if Jpm==0 and Jpmpm == 0 and h == 0:
        return 0, np.array([0,0,0]).reshape((1,3))

    Know = np.pi*np.array([1, 1, 1])
    beta1 = 0.9

    stuff = True
    m = 0
    t = 0

    while stuff:
        print(Know)
        t = t + 1
        g = gradient(Know, np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
        m = beta1 * m + (1-beta1) * g
        print(g)
        Klast = np.copy(Know)
        Know = Know - m
        if (abs(Klast-Know)<1e-8).all():
            break

    Enow = Emin(Know, np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)
    Know = Know.reshape((1,3))
    return -Enow, Know


def check_condensed(Jzz, lamM, M, kappa):
    try:
        if rho_true(Jzz, M, lamM+(680/len(M))**2)[0] < kappa:
            return True
        else:
            return False
    except:
        return False

def run(Jzz, lamM, M, kappa):
    temp = np.copy(lamM)
    a = 1.3
    while True:
        a = a + 0.1
        temp = a * temp
        try:
            if rho_true(Jzz, M, temp)[0] < kappa:
                break
        except:
            pass
    return temp

def findlambda_pi(M, Jzz, kappa, tol, lamM):
    warnings.filterwarnings("error")
    if lamM[0] == 0:
        lamMin = np.zeros(2)
        lamMax = np.ones(2)
    else:
        lamMin = np.copy(lamM)
        if check_condensed(Jzz, lamM, M, kappa):
            lamMax = lamM+(680/len(M))**2
        else:
            lamMax = run(Jzz, lamM+(680/len(M))**2, M, kappa)

    # print(lamMin, lamMax)
    lams = lamMax
    # rhoguess = rho_true(Jzz, M, lams)
    while True:
        lamlast = np.copy(lams)
        lams = (lamMax+lamMin)/2
        try:
            rhoguess = rho_true(Jzz, M, lams)
            for i in range(2):
                if rhoguess[i] - kappa > 0:
                    lamMin[i] = lams[i]
                else:
                    lamMax[i] = lams[i]
            if (abs(lamlast - lams) < 1e-15).all() or ((np.absolute(rhoguess - kappa) <= tol).all()):
                break
        except:
            lamMin = lams
        # print(lams, lamMax-lamMin, rhoguess)

    # print(lams)
    warnings.resetwarnings()
    return lams



def findlambda_pi_scipy(M, Jzz, kappa, tol, lamM):
    if check_condensed(Jzz, lamM, M, kappa):
        lamMax = lamM+(1000/len(M))**2
    else:
        lamMax = 6*lamM[0]
    sol = root_scalar(rho_true_zeroed, args=(Jzz, M, kappa), method='brentq', bracket=(lamM[0], lamMax[0]))
    return sol.root * np.ones(2)

#region Mean field calculation
def chiCal(lams, M, K, Jzz):
    E, V = E_pi_fixed(lams, M)
    E = np.sqrt(2 * Jzz * E)
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jlk->ijl', K, NNminus)
    ffactB = np.exp(1j * ffact)
    A = contract('iab, ijl,jka, lkb->ikjl', green[:, 8:12, 0:4], ffactB, piunitcell, piunitcell)
    M1 = np.mean(A, axis=0)
    chi = M1[0, 0, 3]
    chi0 = np.conj(M1[0, 0, 0])
    return chi, chi0

def xiCal(lams, M, K, Jzz, ns):
    E, V = E_pi_fixed(lams, M)
    E = np.sqrt(2 * Jzz * E)
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jk->ij', K, NN)
    ffactA = np.exp(1j * ffact)

    M1 = np.mean(contract('ika, ij,jka->ikj', green[:, 0:4, 4:8], ffactA, piunitcell), axis=0)

    M1 = M1[0, 0]
    return np.real(M1)

def calmeanfield(lams, M, K, Jzz, ns):
    chi, chi0 = chiCal(lams, M, K, Jzz)
    return chi, chi0, xiCal(lams, M, K, Jzz, ns)

# endregion

#region Mean Field Calculation Condensed
def chiCalC(rhos, K):

    ffact = contract('ik,jlk->ijl', K, NNminus)
    ffactB = np.exp(1j * ffact)
    A = contract('a, b, ijl,jka, lkb->ikjl', rhos[0:4], rhos[0:4], ffactB, piunitcell, piunitcell)
    M1 = np.mean(A, axis=0)
    chi = M1[0, 0, 3]
    chi0 = np.conj(M1[0, 0, 0])
    return chi, chi0

def xiCalC(rhos, K):

    ffact = contract('ik,jk->ij', K, NN)
    ffactA = np.exp(1j * ffact)

    M1 = np.mean(contract('k, a, ij,jka->ikj', rhos[0:4], rhos[4:8], ffactA, piunitcell), axis=0)

    M1 = M1[0, 0]
    return np.real(M1)

def calmeanfieldC(rhos, K):
    chi, chi0 = chiCalC(rhos, K)
    return chi, chi0, xiCalC(rhos, K)

#endregion


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
    A = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])
    mins = A[:,0]
    maxs = A[:,-1]
    for i in range(len(q)):
        temp[i, 0] = np.min(
            np.sqrt(2 * Jzz * E_pi(lams, K - q[i], eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])[:, 0] + mins)
        temp[i, 1] = np.max(
            np.sqrt(2 * Jzz * E_pi(lams, K - q[i], eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi)[0])[:, -1] + maxs)
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
        contract('jl,klj, iab, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced / 4, green[:, 0:4, 0:4], ffactA,
                 piunitcell, piunitcell), axis=0)
    E1B = np.mean(
        contract('jl,klj, iab, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced / 4, green[:, 4:8, 4:8], ffactB,
                 piunitcell, piunitcell), axis=0)

    # print(E1A)
    E1 = np.real(E1A + E1B)

    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(-1j * ffact)
    Emag = np.mean(contract('ku, u, ru, krx, urx->k', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                            np.exp(1j*A_pi), green[:, 0:4, 4:8], piunitcell), axis=0)

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



    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffactA = np.exp(-1j * ffact)
    ffactB = np.exp(1j * ffact)


    E1A = contract('jl,kjl, a, b, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced / 4, rhos[0:4], rhos[0:4], ffactA,
                 piunitcell, piunitcell)
    E1B = contract('jl,kjl, a, b, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced / 4, rhos[4:8], rhos[4:8], ffactB,
                 piunitcell, piunitcell)

    # print(E1A)
    E1 = np.real(np.mean(E1A + E1B))

    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(-1j * ffact)
    Emag = contract('ku, u, ru, r, x, urx->k', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                            np.exp(1j * A_pi), rhos[0:4], rhos[4:8], piunitcell)

    Emag = 2 * np.real(np.mean(Emag))

    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    tempxb = xi[1]
    tempxa = xi[0]
    M1a = contract('jl, kjl, ij, kl, k, x, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa,rhos[0:4], rhos[4:8], piunitcell)
    M1b = contract('jl, kjl, il, kj, k, x, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, tempxa,rhos[0:4], rhos[4:8], piunitcell)
    M2a = contract('jl, kjl, ij, kl, x, k, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),rhos[0:4], rhos[4:8], piunitcell)
    M2b = contract('jl, kjl, il, kj, x, k, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp, ffact, np.conj(tempxb),rhos[0:4], rhos[4:8], piunitcell)
    EAB = 2 * np.real(np.mean(M1a + M1b + M2a + M2b))

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    beta = 1
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, k, k->', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, rhos[0:4], rhos[0:4])
    M2 = np.mean(contract('jl, kjl, ijl, k, b, a, jka, lkb->i', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0, rhos[0:4], rhos[0:4], piunitcell,
                          piunitcell))

    EAA = 2 * np.real(M1 + M2)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * ffact)
    beta = 0
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, k, k->', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, rhos[4:8], rhos[4:8])
    M2 = np.mean(contract('jl, kjl, ijl, k, b, a, jka, lkb->i', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0, rhos[4:8], rhos[4:8], piunitcell,
                          piunitcell))

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
        self.chi = 0.18
        self.xi = 0.5
        self.chi0 = 0.18

        self.minLams = np.zeros(2, dtype=np.double)

        self.BZres = BZres
        self.graphres = graphres
        self.bigB = np.concatenate((genBZ(BZres, 1/2), symK))
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
        self.rhos = np.zeros(8)

    # alpha = 1 for A = -1 for B

    def findLambda(self, MF, minLams):
        return findlambda_pi(MF, self.Jzz, self.kappa, self.tol, minLams)

    def findminLam_old(self):
        return findminLam_old(self.MF, self.Jzz, 1e-10)

    def findminLam(self, chi, chi0, xi):
        minLams, self.qmin = findminLam_scipy(self.MF, self.bigB, self.tol, self.eta, self.Jpm, self.Jpmpm, self.h, self.n,
                                        self.theta, chi, chi0, xi)
        minLams = np.ones(2) * minLams
        K = np.unique(np.concatenate((self.bigB, self.qmin)), axis=0)
        MF = M_pi(K, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, chi0, xi)
        return minLams, K, MF
    def calmeanfield(self, lams, MF, K):

        if self.condensed:
            chic, chi0c, xic = calmeanfieldC(self.rhos, self.qmin)
            chi, chi0, xi = calmeanfield(lams, MF, K, self.Jzz, self.ns)
            return chi + chic, chi0 + chi0c, xi + xic
        else:
            chi, chi0, xi = calmeanfield(lams, MF, K, self.Jzz, self.ns)
        return np.array([chi, chi0, xi])

    # def calmeanfield_con(self, lams, MF, K):
    #     mfs = self.calmeanfield(lams, MF, K)
    #     tol = 1e-7
    #     while True:
    #         mfslast = np.copy(mfs)
    #         MF = M_pi(K, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, mfs[0],  mfs[1],  mfs[2])
    #         mfs = self.calmeanfield(lams, MF, K)
    #         print(mfs, mfslast)
    #         if (abs(mfs+mfslast) < tol).all() or (abs(mfs-mfslast) < tol).all():
    #             break
    #     return mfs


    def solvemeanfield(self, tol=1e-7):
        mfs = np.array([self.chi, self.chi0, self.xi])
        lam, K, MF = self.condensation_check(mfs)
        mfs = self.calmeanfield(lam, MF, K)
        # print(mfs)
        do = not (self.Jpmpm == 0)
        counter = 0
        while do:
            mfslast = np.copy(mfs)
            lam, K, MF = self.condensation_check(mfs)
            mfs = self.calmeanfield(lam, MF, K)
            print(mfs, counter)
            if (abs(mfs+mfslast) < tol).all() or (abs(mfs-mfslast) < tol).all() or counter >= 6:
                break
            counter = counter + 1
        if do:
            lam, K, MF = self.condensation_check(mfs)
        self.chi, self.chi0, self.xi = mfs
        self.lams = lam
        self.MF = MF
        self.bigTemp = K
        self.chi, self.chi0, self.xi = mfs
        self.lams = lam
        self.MF = MF
        self.bigTemp = K
        return 0

    def qvec(self):
        E = \
        E_pi(np.zeros(2), self.bigB, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
             self.xi)[0][:, 0]
        A = np.where(E == E.min())
        self.q = self.bigB[A]

    def ifcondense(self, q, lam, tol=0):
        c = np.array([])
        if self.condensed:
            E, V = self.LV_zero(q, lam)
            E = E[:,0]
            c = np.where(E<=tol)[0]
        c = np.array(c, dtype=int)
        return c

    def low(self):
        E, V = np.linalg.eigh(self.MF)
        cond = np.argmin(E[:, 0])
        return self.bigB[cond], E[cond][0]

    def set_condensed(self, minLams, lams, l):
        A = -minLams[0] + lams[0]
        # B = (2e2 / len(self.bigB)) ** 2
        self.condensed = A < (680/ l) ** 2

    def set_delta(self, K, MF, minLams, lams, l):
        if self.condensed:
            cond = self.ifcondense(K, lams, (680/ l) ** 2)
            MFp = np.delete(MF, cond, axis=0)
            warnings.filterwarnings('error')
            try:
                self.delta = np.sqrt(lams - minLams) * len(self.bigTemp)
            except:
                self.delta = 0
            try:
                self.rhos = np.sqrt(self.kappa - rho_true_site(self.Jzz, MFp, lams))
            except:
                self.rhos = np.zeros(8)
            warnings.resetwarnings()
    def condensation_check(self, mfs):
        chi, chi0, xi = mfs
        start = time.time()
        minLams, K, MF = self.findminLam(chi, chi0, xi)
        end = time.time()
        print('Finding minimum lambda costs ' + str(end-start))
        self.minLams = minLams
        start = time.time()
        lams = self.findLambda(MF, minLams)
        end = time.time()
        print('Finding lambda costs ' + str(end-start))
        l = len(K)
        self.set_condensed(minLams, lams, l)
        self.set_delta(K, MF, minLams, lams, l)
        return lams, K, MF

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


            # print(self.rhos)

    def MFE(self):
        if self.condensed:
            Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi,
                     self.MF,
                     self.lams, self.bigTemp)

            Eq = MFE_condensed(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
                               self.xi, self.MF,
                               self.lams, self.qmin, self.rhos)
            return Ep + Eq
        else:
            Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.MF,
            self.lams, self.bigTemp)
        return Ep

    def MFEs(self, chi, chi0, xi, lams, MF, K):

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
        Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, chi0, xi, MF,
        lams, K)
        return Ep

    def SCE(self, mfs, lams, MF, K):
        tol = 1e-8
        chi, chi0, xi = mfs
        temp = self.MFEs(chi, chi0, xi, lams, MF, K)
        tempChi = (self.MFEs(chi + tol, chi0, xi, lams, MF, K) - temp) / tol
        tempChiC = (self.MFEs(chi + 1j*tol, chi0, xi, lams, MF, K) - temp) / tol
        tempChi0 = (self.MFEs(chi, chi0 + tol, xi, lams, MF, K) - temp) / tol
        tempChi0C = (self.MFEs(chi, chi0 + 1j*tol, xi, lams, MF, K) - temp) / tol
        tempXi = (self.MFEs(chi, chi0, xi + tol, lams, MF, K) - temp) / tol
        return np.array([tempChi+1j*tempChiC, tempChi0+1j*tempChi0C, tempXi])

    def Jacobian(self, mfs, mfslast, f, flast):
        df = f - flast
        dx = mfs - mfslast
        A = contract('i,j->ij', df, 1/dx)
        return A

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

    def print_rho(self):
        minLams, K, MF = self.findminLam(self.chi, self.chi0, self.xi)
        E, V = np.linalg.eigh(MF)
        a = min(E[:,0])
        print(a, a+minLams[0])
        T = np.linspace(minLams, 6*minLams, 50)
        rho = np.zeros(50)
        for i in range(50):
            rho[i] = rho_true(self.Jzz, MF, T[i])[0]
        print(rho_true_zeroed(minLams[0], self.Jzz, MF, self.kappa), rho_true_zeroed(minLams[0] + (1000/len(MF))**2, self.Jzz, MF, self.kappa))
        plt.plot(T, rho)
        plt.show()

    def magnetization(self):
        green = self.green_pi(self.bigTemp)
        ffact = contract('ik, jk->ij', self.bigTemp, NN)
        ffactp = np.exp(-1j * ffact)
        ffactm = np.exp(1j * ffact)

        magp = contract('ij, ika, kj, jka->i', ffactp, green[:, 0:4, 4:8], np.exp(1j * A_pi),
                        piunitcell) / 4
        magm = contract('ij, iak, kj, jka->i', ffactm, green[:, 4:8, 0:4], np.exp(-1j * A_pi),
                        piunitcell) / 4

        con = 0
        if self.condensed:
            ffact = contract('ik, jk->j', self.qmin, NN)
            ffactp = np.exp(-1j * ffact)
            ffactm = np.exp(1j * ffact)

            tempp = contract('j, k, a, kj, jka->j', ffactp, self.rhos[0:4], self.rhos[4:8], np.exp(1j * A_pi),
                            piunitcell) / 4
            tempm = contract('j, a, k, kj, jka->j', ffactm, self.rhos[4:8], self.rhos[0:4], np.exp(-1j * A_pi),
                            piunitcell) / 4

            con = np.mean(tempp+tempm)

        return np.real(np.mean(magp + magm)+con) / 4