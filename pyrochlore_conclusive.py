import time

import matplotlib.pyplot as plt
import warnings

import numpy as np

from misc_helper import *
from flux_stuff import *
from numpy.testing import assert_almost_equal, assert_allclose

#region Hamiltonian Construction
def M_pi_mag_sub_AB(k, h, n, theta, A_pi_here):
    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(-1j * ffact)
    M = contract('ku, u, ru, urx->krx', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                 np.exp(1j*A_pi_here), piunitcell)
    return M


def M_pi_sub_intrahopping_AA(k, alpha, Jpm, A_pi_rs_traced_here):
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    M = contract('jl,kjl,ijl, jka, lkb->iab', notrace, -Jpm * A_pi_rs_traced_here / 4, ffact, piunitcell,
                 piunitcell)
    return M


def M_pi_sub_interhopping_AB(k, alpha, Jpmpm, xi, A_pi_rs_traced_pp_here):
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * neta(alpha) * ffact)
    tempxa = xi[alpha]
    tempxb = xi[1 - alpha]
    M1a = contract('jl, kjl, ij, kl, jkx->ikx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, tempxa, piunitcell)
    M1b = contract('jl, kjl, il, kj, lkx->ikx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, tempxa, piunitcell)
    M2a = contract('jl, kjl, ij, kl, jkx->ixk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(tempxb),
                   piunitcell)
    M2b = contract('jl, kjl, il, kj, lkx->ixk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(tempxb),
                   piunitcell)
    return M1a + M1b + M2a + M2b


def M_pi_sub_pairing_AA(k, alpha, Jpmpm, chi, chi0, A_pi_rs_traced_pp_here):
    d = np.ones(len(k))
    di = np.identity(4)
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    beta = 1 - alpha
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, i, km->ikm', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, tempchi, d, di)
    M2 = contract('jl, kjl, ijl, k, jka, lkb->iba', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0, piunitcell,
                  piunitcell)
    return M1 + M2

def M_pi(k, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    k = contract('ij,jk->ik', k, BasisBZA)

    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])

    dummy = np.zeros((len(k), 4, 4), dtype=np.complex128)

    MAk = M_pi_sub_intrahopping_AA(k, 0, Jpm, A_pi_rs_traced_here)
    MBk = M_pi_sub_intrahopping_AA(k, 1, Jpm, A_pi_rs_traced_here)
    MAnk = M_pi_sub_intrahopping_AA(-k, 0, Jpm, A_pi_rs_traced_here)
    MBnk = M_pi_sub_intrahopping_AA(-k, 1, Jpm, A_pi_rs_traced_here)

    MagAkBk = M_pi_mag_sub_AB(k, h, n, theta, A_pi_here) + M_pi_sub_interhopping_AB(k, 0, Jpmpm, xi, A_pi_rs_traced_pp_here)
    MagBkAk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))
    MagAnkBnk = M_pi_mag_sub_AB(-k, h, n, theta, A_pi_here) + M_pi_sub_interhopping_AB(-k, 0, Jpmpm, xi, A_pi_rs_traced_pp_here)
    MagBnkAnk = np.conj(np.transpose(MagAnkBnk, (0, 2, 1)))

    MAdkAdnk = M_pi_sub_pairing_AA(k, 0, Jpmpm, chi, chi0, A_pi_rs_traced_pp_here)
    MBdkBdnk = M_pi_sub_pairing_AA(k, 1, Jpmpm, chi, chi0, A_pi_rs_traced_pp_here)
    MAnkAk = np.conj(np.transpose(MAdkAdnk, (0, 2, 1)))
    MBnkBk = np.conj(np.transpose(MBdkBdnk, (0, 2, 1)))

    FM = np.block([[MAk, MagAkBk, MAdkAdnk, dummy],
                   [MagBkAk, MBk, dummy, MBdkBdnk],
                   [MAnkAk, dummy, MAnk, MagAnkBnk],
                   [dummy, MBnkBk, MagBnkAnk, MBnk]])
    return FM

#endregion

#region Single K Point Hamiltonian Construction
def M_pi_mag_sub_AB_single(k, h, n, theta, A_pi_here):
    zmag = contract('k,ik->i', n, z)
    ffact = contract('k, jk->j', k, NN)
    ffact = np.exp(-1j * ffact)
    M = contract('u, u, ru, urx->rx', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                 np.exp(1j*A_pi_here), piunitcell)
    return M


def M_pi_sub_intrahopping_AA_single(k, alpha, Jpm, A_pi_rs_traced_here):
    ffact = contract('k, jlk->jl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    M = contract('jl,kjl,jl, jka, lkb->ab', notrace, -Jpm * A_pi_rs_traced_here / 4, ffact, piunitcell,
                 piunitcell)
    return M


def M_pi_sub_interhopping_AB_single(k, alpha, Jpmpm, xi, A_pi_rs_traced_pp_here):
    ffact = contract('k, jk->j', k, NN)
    ffact = np.exp(1j * neta(alpha) * ffact)
    tempxa = xi[alpha]
    tempxb = xi[1 - alpha]
    M1a = contract('jl, kjl, j, kl, jkx->kx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, tempxa, piunitcell)
    M1b = contract('jl, kjl, l, kj, lkx->kx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, tempxa, piunitcell)
    M2a = contract('jl, kjl, j, kl, jkx->xk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(tempxb),
                   piunitcell)
    M2b = contract('jl, kjl, l, kj, lkx->xk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(tempxb),
                   piunitcell)
    return M1a + M1b + M2a + M2b


def M_pi_sub_pairing_AA_single(k, alpha, Jpmpm, chi, chi0, A_pi_rs_traced_pp_here):
    di = np.identity(4)
    ffact = contract('k, jlk->jl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    beta = 1 - alpha
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, km-> km', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, tempchi, di)
    M2 = contract('jl, kjl, jl, k, jka, lkb->ba', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0, piunitcell,
                  piunitcell)
    return M1 + M2

def M_pi_single(k, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])

    dummy = np.zeros((4, 4), dtype=np.complex128)

    MAk = M_pi_sub_intrahopping_AA_single(k, 0, Jpm, A_pi_rs_traced_here)
    MBk = M_pi_sub_intrahopping_AA_single(k, 1, Jpm, A_pi_rs_traced_here)
    MAnk = M_pi_sub_intrahopping_AA_single(-k, 0, Jpm, A_pi_rs_traced_here)
    MBnk = M_pi_sub_intrahopping_AA_single(-k, 1, Jpm, A_pi_rs_traced_here)

    MagAkBk = M_pi_mag_sub_AB_single(k, h, n, theta, A_pi_here) + M_pi_sub_interhopping_AB_single(k, 0, Jpmpm, xi, A_pi_rs_traced_pp_here)
    MagBkAk = np.conj(np.transpose(MagAkBk))
    MagAnkBnk = M_pi_mag_sub_AB_single(-k, h, n, theta, A_pi_here) + M_pi_sub_interhopping_AB_single(-k, 0, Jpmpm, xi, A_pi_rs_traced_pp_here)
    MagBnkAnk = np.conj(np.transpose(MagAnkBnk))

    MAdkAdnk = M_pi_sub_pairing_AA_single(k, 0, Jpmpm, chi, chi0, A_pi_rs_traced_pp_here)
    MBdkBdnk = M_pi_sub_pairing_AA_single(k, 1, Jpmpm, chi, chi0, A_pi_rs_traced_pp_here)
    MAnkAk = np.conj(np.transpose(MAdkAdnk))
    MBnkBk = np.conj(np.transpose(MBdkBdnk))

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


def E_pi(k, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    M = M_pi(k, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    E, V = np.linalg.eigh(M)
    return E, V

def I3_integrand(E, lams, Jzz):
    E = np.sqrt(2*Jzz*(E+lams[0]))
    Ep = Jzz / E
    return np.mean(Ep,axis=1)


def rho_true(E, lams, Jzz, weights):
    return integrate_fixed(I3_integrand, weights,E, lams, Jzz)


def I3_integrand_old(k, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here,
             A_pi_rs_traced_pp_here):
    M = M_pi(k, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here,
             A_pi_rs_traced_pp_here) + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eig(M)
    Vt = np.real(contract('ijk,ijk->ijk', V, np.conj(V)))
    Ep = contract('ijk, ik->ij', Vt, Jzz / np.sqrt(2 * Jzz * E))
    return np.mean(Ep,axis=1)

def rho_true_old(pts, weights, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here,
             A_pi_rs_traced_pp_here):
    return integrate(I3_integrand_old, pts, weights, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here,
             A_pi_rs_traced_pp_here)


# def rho_true_adaptive(tol, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
#     return adaptive_gauss_quadrature_3d(I3_integrand, 0, 1, 0, 1, 0, 1, tol, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)

def Emin(q, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    q = q.reshape((1,3))
    return E_pi(q, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0][0,0]


def gradient(k, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    kx, ky, kz = k
    step = 1e-8
    fx = (Emin(np.array([kx + step, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - Emin(
        np.array([kx, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step
    fy = (Emin(np.array([kx, ky + step, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - Emin(
        np.array([kx, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step
    fz = (Emin(np.array([kx, ky, kz + step]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - Emin(
        np.array([kx, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step
    return np.array([fx, fy, fz])

def hessian(k, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    kx, ky, kz = k
    step = 1e-8

    fxx = (Emin(np.array([kx + step, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx - step, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    fxy = (Emin(np.array([kx, ky + step, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx - step, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    fxz = (Emin(np.array([kx, ky, kz + step]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx - step, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    fyy = (Emin(np.array([kx, ky+ step, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx, ky - step, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    fyz = (Emin(np.array([kx, ky, kz+ step]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx, ky - step, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    fzz = (Emin(np.array([kx, ky, kz + step]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx, ky, kz - step]), Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    return np.array([[fxx, fxy, fxz],[fxy, fyy, fyz],[fxz, fyz, fzz]])

def findminLam(M, K, tol, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, BZres):
    if Jpm==0 and Jpmpm == 0 and h == 0:
        return 0, np.array([0,0,0]).reshape((1,3))
    warnings.filterwarnings("error")
    E, V = np.linalg.eigh(M)
    E = np.around(E[:,0], decimals=16)
    Em = E.min()
    dex = np.where(E==Em)
    Know = np.unique(np.around(K[dex], decimals=15), axis=0)
    # Know = Know + (np.random.rand(Know.shape[0], Know.shape[1])-1/2) / (2*BZres)

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
            if not init:
                gradlen = gradient(Know[i], Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - gradient(Klast,
                        np.zeros(2),Jpm,Jpmpm, h, n,theta,chi,chi0,xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
                try:
                    step = abs(np.dot(Know[i] - Klast, gradlen)) / np.linalg.norm(gradlen) ** 2
                except:
                    step = 0

            Klast = np.copy(Know[i])
            Know[i] = Know[i] - step * gradient(Know[i], Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
            Elast = np.copy(Enow[i])
            Enow[i] = Emin(Know[i], Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
            init = False
            if abs(Elast-Enow[i])<1e-12:
                stuff = False
    warnings.resetwarnings()

    a = np.argmin(Enow)
    Know = np.mod(Know[a], 2*np.pi).reshape((1,3))
    # Know = np.where(Know>np.pi, Know-2*np.pi, Know)
    return -Enow[a], Know

def findminLam_scipy(M, K, tol, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, BZres):
    if Jpm==0 and Jpmpm == 0 and h == 0:
        return 1/8, np.array([0,0,0]).reshape((1,3))

    E, V = np.linalg.eigh(M)
    E = np.around(E[:,0], decimals=16)
    Em = E.min()
    dex = np.where(E==Em)
    Know = np.unique(np.mod(K[dex],1), axis=0)


    if Know.shape == (3,):
        Know = Know.reshape(1,3)

    if len(Know) > 8:
        Know = Know[0:8]

    Enow = np.zeros(len(Know))

    for i in range(len(Know)):
        res = minimize(Emin, x0=Know[i], args=(Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here,
                                               A_pi_rs_traced_here, A_pi_rs_traced_pp_here),
                       method='Nelder-Mead', bounds=((Know[i,0]-1/BZres, Know[i,0]+1/BZres), (Know[i,1]-1/BZres,Know[i,1]+1/BZres), (Know[i,2]-1/BZres,Know[i,2]+1/BZres)))
        Know[i] = np.array(res.x)
        Enow[i] = res.fun
    Enowm = Enow.min()
    dex = np.where(Enow==Enowm)
    # print(Know, Enow)
    Know = np.unique(np.mod(Know[dex], 1),axis=0)
    Know = contract('ij,jk->ik', Know, BasisBZA)
    if Know.shape == (3,):
        Know = Know.reshape(1,3)

    return -Enowm, Know

def findlambda_pi(kappa, tol, E, lamM, Jzz, weights):
    warnings.filterwarnings("error")
    if lamM[0] == 0:
        lamMin = np.zeros(2)
        lamMax = np.ones(2)
    else:
        lamMin = np.copy(lamM)
        lamMax = 10*np.ones(2)
    lams = lamMax
    count = 0
    while True:
        lamlast = np.copy(lams)
        lams = (lamMax+lamMin)/2
        try:
            rhoguess = rho_true(E, lams, Jzz, weights)
            # rhoguess = rho_true_old(k, weights, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0,xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
            error = rhoguess-kappa
            # print(lams, lamMin, lamMax, rhoguess,error)
            if error > 0:
                lamMin = lams
            else:
                lamMax = lams
            if (abs(lamlast - lams) < 1e-15).all() or ((np.absolute(rhoguess - kappa) <= tol).all()):
                break
        except:
            lamMin = lams
            # print(lams, rhoguess)
    warnings.resetwarnings()
    return lams

#region Mean field calculation

def chi_integrand(k, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):

    M = M_pi(k, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    E, V = E_pi_fixed(lams, M)
    E = np.sqrt(2 * Jzz * E)
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jlk->ijl', k, NNminus)
    ffactB = np.exp(1j * ffact)
    A = contract('iab, ijl,jka, lkb->kjil', green[:, 8:12, 0:4], ffactB, piunitcell, piunitcell)

    return A

def chiCal(BZres, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, pts, weights):
    M1 = integrate(chi_integrand, pts, weights, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    chi = M1[0, 0, 3]
    chi0 = np.conj(M1[0, 0, 0])
    chi = chi * np.sign(chi)
    chi0 = chi0 * np.sign(chi0)
    return chi, chi0

def xi_integrand(k, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):

    M = M_pi(k, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    E, V = E_pi_fixed(lams, M)
    E = np.sqrt(2 * Jzz * E)
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jk->ij', k, NN)
    ffactA = np.exp(1j * ffact)

    A = contract('ika, ij,jka->kij', green[:, 0:4, 4:8], ffactA, piunitcell)

    return A
def xiCal(BZres, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, pts, weights):
    M = integrate(xi_integrand, pts, weights, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    M1 = M[0, 0]
    return np.real(np.abs(M1))

def calmeanfield(BZres, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, pts, weights):
    chi, chi0 = chiCal(BZres, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,pts, weights)
    return chi, chi0, xiCal(BZres, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,pts, weights)

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
    return np.real(np.abs(M1))

def calmeanfieldC(rhos, K):
    chi, chi0 = chiCalC(rhos, K)
    return chi, chi0, xiCalC(rhos, K)

#endregion


# graphing BZ

def dispersion_pi(lams, k, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    temp = np.sqrt(2 * Jzz * E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])
    return temp


def calDispersion(lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    dGammaX = dispersion_pi(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dXW = dispersion_pi(lams, XW, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dWK = dispersion_pi(lams, WK, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dKGamma = dispersion_pi(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dGammaL = dispersion_pi(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dLU = dispersion_pi(lams, LU, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dUW = dispersion_pi(lams, UW, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)

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
def minCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    temp = np.zeros(len(q))
    mins = np.sqrt(2 * Jzz * E_pi(K, lams, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])[:, 0]
    for i in range(len(q)):
        temp[i] = np.min(
            np.sqrt(2 * Jzz * E_pi(K - q[i], lams, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])[:, 0] + mins)
    return temp


# @nb.njit(parallel=True, cache=True)
def maxCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    temp = np.zeros(len(q))
    maxs = np.sqrt(2 * Jzz * E_pi(K, lams, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])[:, -1]
    for i in range(len(q)):
        temp[i] = np.max(
            np.sqrt(2 * Jzz * E_pi(K - q[i], lams, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])[:, -1] + maxs)
    return temp


def minMaxCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    temp = np.zeros((len(q), 2))
    Ek = np.sqrt(2 * Jzz * E_pi(K, lams, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])
    mins = Ek[:,0]
    maxs = Ek[:,-1]
    for i in range(len(q)):
        tt = np.sqrt(2 * Jzz * E_pi(K - q[i], lams, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])
        temp[i, 0] = np.min(tt[:, 0] + mins)
        temp[i, 1] = np.max(tt[:, -1] + maxs)
    return temp


def loweredge(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    dGammaX = minCal(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dXW = minCal(lams, XW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dWK = minCal(lams, WK, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dLU = minCal(lams, LU, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dUW = minCal(lams, UW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)

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


def upperedge(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    dGammaX = maxCal(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dXW = maxCal(lams, XW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dWK = maxCal(lams, WK, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dKGamma = maxCal(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dGammaL = maxCal(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dLU = maxCal(lams, LU, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dUW = maxCal(lams, UW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)

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
    green = Jzz / E
    green = contract('ilk, ijk, ik->ijl', V, np.conj(V), green)
    return green

def green_pi_branch(E, V, Jzz):
    Vt = contract('ijk, ilk->iklj', V, np.conj(V))
    green = Jzz / E
    green = contract('ikjl, ik->ikjl', Vt, green)
    return green



def Encompassing_integrand(q, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])

    M = M_pi(q, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(M)
    E = np.sqrt(2 * Jzz * E)

    EQ = 2*np.real(np.mean(E,axis=1))
    green = green_pi(E, V, Jzz)

    k = contract('ij,jk->ik', q, BasisBZA)
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffactA = np.exp(-1j * ffact)
    E1 = np.real(contract('jl,klj, iab, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced_here / 4, green[:,0:4,0:4], ffactA,
                 piunitcell, piunitcell))

    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(-1j * ffact)

    Emag = np.real(contract('ku, u, ru, krx, urx->k', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                            np.exp(1j*A_pi_here), green[:, 0:4, 4:8], piunitcell))

    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    tempxb = xi[1]
    tempxa = xi[0]

    M1a = contract('jl, kjl, ij, kl, ikx, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, tempxa,
                           green[:, 0:4, 4:8], piunitcell)
    M1b = contract('jl, kjl, il, kj, ikx, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, tempxa,
                           green[:, 0:4, 4:8], piunitcell)
    M2a = contract('jl, kjl, ij, kl, ixk, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(tempxb),
                 green[:, 0:4, 4:8], piunitcell)
    M2b = contract('jl, kjl, il, kj, ixk, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(tempxb),
                 green[:, 0:4, 4:8], piunitcell)

    EAB = np.real(M1a + M1b + M2a + M2b)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    tempchi = chi[1]
    tempchi0 = chi0[1]

    M1 = contract('jl, kjl, kjl, ikk->i', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, tempchi, green[:, 0:4, 8:12])

    M2 = contract('jl, kjl, ijl, k, iba, jka, lkb->i', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0,
                          green[:, 0:4, 8:12], piunitcell, piunitcell)

    EAA = np.real(M1+M2)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * ffact)
    tempchi = chi[0]
    tempchi0 = chi0[0]

    M1 = contract('jl, kjl, kjl, ikk->i', notrace, Jpmpm * A_pi_rs_traced_pp / 8, tempchi, green[:, 4:8, 12:16])

    M2 = contract('jl, kjl, ijl, k, iba, jka, lkb->i', notrace, Jpmpm * A_pi_rs_traced_pp / 8, ffact, tempchi0,
                          green[:, 4:8, 12:16], piunitcell, piunitcell)
    EBB = np.real(M1+ M2)
    Etot = EQ + Emag + E1 + EAB + EAA + EBB
    return Etot

def MFE(Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, lams, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, BZres, kappa, pts, weights):
    Eall = integrate(Encompassing_integrand, pts, weights, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    # print(EQ/4, E1/4, Emag/4, EAB/4, EAA/4, EBB/4, E/4)
    return Eall/4

def MFE_condensed(Jpm, Jpmpm, h, n, theta, chi, chi0, xi, k, rhos, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])



    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffactA = np.exp(-1j * ffact)
    ffactB = np.exp(1j * ffact)


    E1A = contract('jl,kjl, a, b, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced_here / 4, rhos[0:4], rhos[0:4], ffactA,
                 piunitcell, piunitcell)
    E1B = contract('jl,kjl, a, b, ijl, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced_here / 4, rhos[4:8], rhos[4:8], ffactB,
                 piunitcell, piunitcell)

    # print(E1A)
    E1 = np.real(np.mean(E1A + E1B))

    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(-1j * ffact)
    Emag = contract('ku, u, ru, r, x, urx->k', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                            np.exp(1j * A_pi_here), rhos[0:4], rhos[4:8], piunitcell)

    Emag = 2 * np.real(np.mean(Emag))

    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    tempxb = xi[1]
    tempxa = xi[0]
    M1a = contract('jl, kjl, ij, kl, k, x, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, tempxa,rhos[0:4], rhos[4:8], piunitcell)
    M1b = contract('jl, kjl, il, kj, k, x, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, tempxa,rhos[0:4], rhos[4:8], piunitcell)
    M2a = contract('jl, kjl, ij, kl, x, k, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(tempxb),rhos[0:4], rhos[4:8], piunitcell)
    M2b = contract('jl, kjl, il, kj, x, k, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(tempxb),rhos[0:4], rhos[4:8], piunitcell)
    EAB = 2 * np.real(np.mean(M1a + M1b + M2a + M2b))

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    beta = 1
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, k, k->', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, tempchi, rhos[0:4], rhos[0:4])
    M2 = np.mean(contract('jl, kjl, ijl, k, b, a, jka, lkb->i', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0, rhos[0:4], rhos[0:4], piunitcell,
                          piunitcell))

    EAA = 2 * np.real(M1 + M2)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * ffact)
    beta = 0
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = contract('jl, kjl, kjl, k, k->', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, tempchi, rhos[4:8], rhos[4:8])
    M2 = np.mean(contract('jl, kjl, ijl, k, b, a, jka, lkb->i', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0, rhos[4:8], rhos[4:8], piunitcell,
                          piunitcell))

    EBB = 2 * np.real(M1 + M2)

    E = Emag + E1 + EAB + EAA + EBB
    # print(E1/4, Emag/4, EAB/4, EAA/4, EBB/4)
    return E / 4



class piFluxSolver:
    def __init__(self, Jxx, Jyy, Jzz, theta=0, h=0, n=np.array([0, 0, 0]), kappa=2, lam=2, BZres=20, graphres=20,
                 ns=1, tol=1e-12, flux=np.zeros(4), intmethod=gauss_quadrature_3D_pts):
        self.intmethod = intmethod
        self.Jzz = Jzz
        self.Jpm = -(Jxx + Jyy) / 4
        self.Jpmpm = (Jxx - Jyy) / 4
        self.theta = theta
        self.kappa = kappa
        self.tol = tol
        self.lams = np.array([lam, lam], dtype=np.double)
        self.ns = ns
        self.h = h
        self.n = n
        self.chi = 0.18
        self.xi = 0.5
        self.chi0 = 0.18

        if (n == h110).all():
            self.A_pi_here = constructA_pi_110(flux)
        elif (n == h111).all():
            self.A_pi_here = constructA_pi_111(flux)
        elif (n == h001).all():
            self.A_pi_here = constructA_pi_001(flux)

        self.pts, self.weights = self.intmethod(0, 1, 0, 1, 0, 1, BZres)

        self.minLams = np.zeros(2, dtype=np.double)

        self.BZres = BZres
        self.graphres = graphres

        self.q = np.nan
        self.qmin = np.empty(3)
        self.qmin[:] = np.nan
        self.condensed = False
        self.delta = np.zeros(2)
        self.rhos = np.zeros(8)

        self.A_pi_rs_traced_here = np.zeros((4, 4, 4), dtype=np.complex128)

        for i in range(4):
            for j in range(4):
                for k in range(4):
                    self.A_pi_rs_traced_here[i, j, k] = np.exp(1j * (self.A_pi_here[i, j] - self.A_pi_here[i, k]))

        self.A_pi_rs_traced_pp_here = np.zeros((4, 4, 4), dtype=np.complex128)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    self.A_pi_rs_traced_pp_here[i, j, k] = np.exp(1j * (self.A_pi_here[i, j] + self.A_pi_here[i, k]))


        self.A_pi_rs_rsp_here = np.zeros((4, 4, 4, 4), dtype=np.complex128)

        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        self.A_pi_rs_rsp_here[i, j, k, l] = np.exp(1j * (self.A_pi_here[i, k] - self.A_pi_here[j, l]))

        self.A_pi_rs_rsp_pp_here = np.zeros((4, 4, 4, 4), dtype=np.complex128)

        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        self.A_pi_rs_rsp_pp_here[i, j, k, l] = np.exp(1j * (self.A_pi_here[i, k] + self.A_pi_here[j, l]))
        self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
                       self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        self.E, self.V = np.linalg.eigh(self.MF)

    def findLambda(self):
        return findlambda_pi(self.kappa,self.tol, self.E, self.minLams, self.Jzz, self.weights)

    def findminLam(self, chi, chi0, xi):
        minLams, self.qmin = findminLam_scipy(self.MF, self.pts, self.tol, self.Jpm, self.Jpmpm, self.h, self.n,
                                        self.theta, chi, chi0, xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.BZres)
        self.minLams = np.ones(2) * minLams
        return minLams

    def rho(self,lam):
        return rho_true(self.E, lam, self.Jzz, self.weights)
    def calmeanfield(self, lams):
        if self.condensed:
            chic, chi0c, xic = calmeanfieldC(self.rhos, self.qmin)
            chi, chi0, xi = calmeanfield(self.BZres, lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta,
                                         self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here,self.pts, self.weights)
            return chi + chic, chi0 + chi0c, xi + xic
        else:
            chi, chi0, xi = calmeanfield(self.BZres, lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta,
                                         self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here,self.pts, self.weights)
            return np.array([chi, chi0, xi])

    def solvemeanfield(self, tol=1e-15):
        mfs = np.array([self.chi, self.chi0, self.xi])
        self.condensation_check(mfs)
        mfs = self.calmeanfield(self.lams)
        do = not (self.Jpmpm == 0)
        counter = 0
        while do:
            mfslast = np.copy(mfs)
            self.condensation_check(mfs)
            mfs = self.calmeanfield(self.lams)
            print(mfs, self.lams, self.minLams)
            if (abs(mfs-mfslast) < tol).all() or counter >= 30:
                break
            counter = counter + 1
        if do:
            self.condensation_check(mfs)
        self.chi, self.chi0, self.xi = mfs
        return 0

    def ifcondense(self, q, lam, tol=0):
        c = np.array([])
        if self.condensed:
            E, V = self.LV_zero(q, lam)
            E = E[:,0]
            c = np.where(E<=tol)[0]
        self.toignore = np.array(c, dtype=int)

    def low(self):
        E, V = np.linalg.eigh(self.MF)
        cond = np.argmin(E[:, 0])
        return self.bigB[cond], E[cond][0]

    def set_condensed(self):
        self.condensed = self.rho(self.minLams+5e-15) < self.kappa

    def set_delta(self):
        warnings.filterwarnings('error')
        try:
            self.rhos = np.sqrt(self.kappa - self.rho(self.minLams))*np.ones(8)
            self.delta = np.sqrt(self.Jzz/2)/self.rhos**2
        except:
            self.rhos = np.zeros(8)
            self.delta = np.zeros(2)
        warnings.resetwarnings()

    def condensation_check(self, mfs):
        chi, chi0, xi = mfs
        self.findminLam(chi, chi0, xi)
        self.set_condensed()
        if self.condensed:
            self.set_delta()
            self.lams=self.minLams
        else:
            self.lams = self.findLambda()
    def M_true(self, k):
        return M_pi(k, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def E_compute(self, k):
        M = self.M_true(k) + np.diag(np.repeat(np.repeat(self.lams, 4), 2))
        E, V = np.linalg.eigh(M)
        return np.mean(np.sqrt(2 * self.Jzz * E), axis=1)

    def E_pi(self, k):
        return np.mean(np.sqrt(2 * self.Jzz *
                               (self.lams[0]+E_pi(k, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi,
                            self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)[0])),axis=1)

    def dispersion(self, k):
        return dispersion_pi(self.lams, k, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta,
                             self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def LV_zero(self, k, lam=np.zeros(2)):
        if np.any(lam == 0):
            lam = self.lams
        return E_pi(k, lam, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
    def GS(self):
        # return np.dot(self.E_compute(self.pts), self.weights) - self.kappa*self.lams[0]
        return np.dot(np.sqrt(2*self.Jzz*(np.mean(self.E, axis=1)+self.lams[0])), self.weights) - self.kappa*self.lams[0]


    def MFE(self):
        if self.condensed:
            Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi,
                     self.lams, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.BZres, self.kappa, self.pts, self.weights)

            Eq = MFE_condensed(self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
                               self.xi, self.qmin, self.rhos, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
            return Ep + Eq
        else:
            Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi,
            self.lams, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.BZres, self.kappa, self.pts, self.weights)
        return Ep

    def graph(self, show):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi,
                      self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        if show:
            plt.show()

    def minCal(self, K):
        return minCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.bigB, self.theta,
                      self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def maxCal(self, K):
        return maxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.bigB, self.theta,
                      self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def minMaxCal(self, K):
        return minMaxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.bigB, self.theta,
                         self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def EMAX(self):
        return np.sqrt(2 * self.Jzz * EMAX(self.MF, self.lams))

    def TWOSPINON_GAP(self, k):
        return np.min(self.minCal(k))

    def TWOSPINON_MAX(self, k):
        return np.max(self.maxCal(k))

    def TWOSPINON_DOMAIN(self, k):
        A = self.minMaxCal(k)
        return np.min(A[:,0]), np.max(A[:,1])

    def graph_loweredge(self, show):
        loweredge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.bigB, self.theta, self.chi,
                  self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        if show:
            plt.show()

    def graph_upperedge(self, show):
        upperedge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.bigB, self.theta, self.chi,
                  self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        if show:
            plt.show()

    def green_pi(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi(E, V, self.Jzz)

    def green_pi_branch(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi_branch(E, V, self.Jzz), E

    def mag_integrand(self, k):
        M = M_pi(k, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here,
                 self.A_pi_rs_traced_pp_here) + np.diag(np.repeat(np.repeat(self.lams, 4), 2))
        E, V = np.linalg.eigh(M)
        E = np.sqrt(2 * self.Jzz * E)
        green = green_pi(E, V, self.Jzz)

        ffact = contract('ik, jk->ij', k, NN)
        ffact = np.exp(-1j * ffact)

        magp = 2 * np.real(contract('ku, ru, krx, urx->k', ffact,
                             np.exp(1j * self.A_pi_here), green[:, 0:4, 4:8], piunitcell))
        return magp


    def magnetization(self):
        mag = integrate(self.mag_integrand, self.pts, self.weights)
        con = 0
        if self.condensed:
            ffact = contract('ik, jk->ij', self.qmin, NN)
            ffactp = np.exp(-1j * ffact)
            ffactm = np.exp(1j * ffact)

            tempp = contract('ij, k, a, kj, jka->i', ffactp, self.rhos[0:4], self.rhos[4:8], np.exp(1j * self.A_pi_here),
                            piunitcell) / 4
            tempm = contract('ij, a, k, kj, jka->i', ffactm, self.rhos[4:8], self.rhos[0:4], np.exp(-1j * self.A_pi_here),
                            piunitcell) / 4

            con = np.mean(tempp+tempm)

        return  np.real(mag+con)