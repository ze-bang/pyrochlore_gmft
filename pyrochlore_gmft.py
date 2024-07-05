import matplotlib.pyplot as plt
import warnings

import numpy as np

from misc_helper import *
from flux_stuff import *
import time

#region Hamiltonian Construction
def M_pi_mag_sub_AB(k, h, n, theta, A_pi_here, unitcell=piunitcell):
    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    M = contract('ku, u, ru, urx->krx', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                 np.exp(1j*A_pi_here), unitcell)
    return M

def M_pi_sub_intrahopping_AA(k, alpha, Jpm, A_pi_rs_traced_here, unitcell=piunitcell):
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * neta(alpha) * ffact)
    M = contract('jl,kjl,ijl, jka, lkb->iab', notrace, -Jpm * A_pi_rs_traced_here / 4, ffact, unitcell,
                 unitcell)
    return M


def M_pi_sub_interhopping_AB(k, alpha, Jpmpm, xi, A_pi_rs_traced_pp_here, unitcell=piunitcell):
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * neta(alpha) * ffact)
    M1a = contract('jl, kjl, ij, kl, jkx->ikx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, xi, unitcell)
    M1b = contract('jl, kjl, il, kj, lkx->ikx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, xi, unitcell)
    return M1a + M1b


def M_pi_sub_pairing_AA(k, alpha, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell=piunitcell):
    d = np.ones(len(k))
    di = np.identity(unitcell.shape[1])
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
    tempchi0 = chi[1-alpha, :,0,0]
    M1 = contract('jl, kjl, kjl, i, km->ikm', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, chi[1-alpha], d, di)
    M2 = contract('jl, kjl, ijl, k, jka, lkb->iba', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0, unitcell,
                  unitcell)
    return M1 + M2

def M_pi(k, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
         unitcell=piunitcell):

    k = contract('ij,jk->ik', k, BasisBZA)
    size = len(A_pi_here)


    MAk = M_pi_sub_intrahopping_AA(k, 0, Jpm, A_pi_rs_traced_here, unitcell)
    MBk = M_pi_sub_intrahopping_AA(k, 1, Jpm, A_pi_rs_traced_here, unitcell)
    MAnk = M_pi_sub_intrahopping_AA(-k, 0, Jpm, A_pi_rs_traced_here, unitcell)
    MBnk = M_pi_sub_intrahopping_AA(-k, 1, Jpm, A_pi_rs_traced_here, unitcell)



    MagAkBk = M_pi_mag_sub_AB(k, h, n, theta, A_pi_here, unitcell)
    MagBkAk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))

    if Jpmpm == 0:
        FM = np.block([[MAk, MagAkBk],
                       [MagBkAk, MBk]])
    else:
        dummy = np.zeros((len(k), size, size), dtype=np.complex128)

        MagAkBk = MagAkBk + M_pi_sub_interhopping_AB(k, 0, Jpmpm, xi, A_pi_rs_traced_pp_here, unitcell)
        MagBkAk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))
        MagAnkBnk = M_pi_mag_sub_AB(-k, h, n, theta, A_pi_here, unitcell) + M_pi_sub_interhopping_AB(-k, 0, Jpmpm, xi, A_pi_rs_traced_pp_here, unitcell)
        MagBnkAnk = np.conj(np.transpose(MagAnkBnk, (0, 2, 1)))

        MAdkAdnk = M_pi_sub_pairing_AA(k, 0, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell)
        MBdkBdnk = M_pi_sub_pairing_AA(k, 1, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell)
        MAnkAk = np.conj(np.transpose(MAdkAdnk, (0, 2, 1)))
        MBnkBk = np.conj(np.transpose(MBdkBdnk, (0, 2, 1)))

        # print(MAk.shape[1], MagAkBk.shape[1],MAdkAdnk.shape[1],dummy.shape[1])
        FM = np.block([[MAk, MagAkBk, MAdkAdnk, dummy],
                       [MagBkAk, MBk, dummy, MBdkBdnk],
                       [MAnkAk, dummy, MAnk, MagBnkAnk],
                       [dummy, MBnkBk, MagAnkBnk, MBnk]])

    return FM

#endregion

#region E_pi
def E_pi_fixed(lams, M):
    M = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
    E, V = np.linalg.eigh(M)
    return E, V


def E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
         unitcell=piunitcell):
    M = M_pi(k, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitcell)
    M = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
    E, V = np.linalg.eigh(M)
    return [E, V]

#endregion

#region find lambda



def I3_integrand(E, lams, Jzz):
    E = np.sqrt(2*Jzz*(E+lams[0]))
    Ep = Jzz / E
    return np.mean(Ep,axis=1)

def I3_integrand_site(E, V, lams, Jzz):
    E = np.sqrt(2*Jzz*(E+lams[0]))
    Ep = contract('ijk, ijk, ik->ij', V, np.conj(V), Jzz/E)
    return Ep


def rho_true(weights, E, lams, Jzz):
    return integrate_fixed(I3_integrand, weights, E, lams, Jzz)

def rho_true_site(weights, E, V, lams, Jzz):
    return integrate_fixed(I3_integrand_site, weights, E, V, lams, Jzz)

#endregion

#region gradient find minLam
def Emin(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitcell):
    k = k.reshape((1,3))
    return E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitcell)[0][0,0]

def Emins(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    return E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0][:, 0]



def gradient(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    kx, ky, kz = k
    step = 1e-8
    fx = (Emin(np.array([kx + step, ky, kz]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
               A_pi_rs_traced_pp_here, piunitcell) - Emin(np.array([kx, ky, kz]), lams, Jpm, Jpmpm, h, n, theta, chi,
                                                          xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
                                                          piunitcell)) / step
    fy = (Emin(np.array([kx, ky + step, kz]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
               A_pi_rs_traced_pp_here, piunitcell) - Emin(np.array([kx, ky, kz]), lams, Jpm, Jpmpm, h, n, theta, chi,
                                                          xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
                                                          piunitcell)) / step
    fz = (Emin(np.array([kx, ky, kz + step]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
               A_pi_rs_traced_pp_here, piunitcell) - Emin(np.array([kx, ky, kz]), lams, Jpm, Jpmpm, h, n, theta, chi,
                                                          xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
                                                          piunitcell)) / step
    return np.array([fx, fy, fz])

def hessian(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    kx, ky, kz = k
    step = 1e-8

    fxx = (Emin(np.array([kx + step, ky, kz]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                A_pi_rs_traced_pp_here, piunitcell) - 2 * Emin(np.array([kx, ky, kz]), lams, Jpm, Jpmpm, h, n, theta,
                                                               chi, xi, A_pi_here, A_pi_rs_traced_here,
                                                               A_pi_rs_traced_pp_here, piunitcell)
           + Emin(np.array([kx - step, ky, kz]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, piunitcell)) / step ** 2
    fxy = (Emin(np.array([kx, ky + step, kz]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                A_pi_rs_traced_pp_here, piunitcell) - 2 * Emin(np.array([kx, ky, kz]), lams, Jpm, Jpmpm, h, n, theta,
                                                               chi, xi, A_pi_here, A_pi_rs_traced_here,
                                                               A_pi_rs_traced_pp_here, piunitcell)
           + Emin(np.array([kx - step, ky, kz]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, piunitcell)) / step ** 2
    fxz = (Emin(np.array([kx, ky, kz + step]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                A_pi_rs_traced_pp_here, piunitcell) - 2 * Emin(np.array([kx, ky, kz]), lams, Jpm, Jpmpm, h, n, theta,
                                                               chi, xi, A_pi_here, A_pi_rs_traced_here,
                                                               A_pi_rs_traced_pp_here, piunitcell)
           + Emin(np.array([kx - step, ky, kz]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, piunitcell)) / step ** 2
    fyy = (Emin(np.array([kx, ky + step, kz]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                A_pi_rs_traced_pp_here, piunitcell) - 2 * Emin(np.array([kx, ky, kz]), lams, Jpm, Jpmpm, h, n, theta,
                                                               chi, xi, A_pi_here, A_pi_rs_traced_here,
                                                               A_pi_rs_traced_pp_here, piunitcell)
           + Emin(np.array([kx, ky - step, kz]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, piunitcell)) / step ** 2
    fyz = (Emin(np.array([kx, ky, kz + step]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                A_pi_rs_traced_pp_here, piunitcell) - 2 * Emin(np.array([kx, ky, kz]), lams, Jpm, Jpmpm, h, n, theta,
                                                               chi, xi, A_pi_here, A_pi_rs_traced_here,
                                                               A_pi_rs_traced_pp_here, piunitcell)
           + Emin(np.array([kx, ky - step, kz]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, piunitcell)) / step ** 2
    fzz = (Emin(np.array([kx, ky, kz + step]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                A_pi_rs_traced_pp_here, piunitcell) - 2 * Emin(np.array([kx, ky, kz]), lams, Jpm, Jpmpm, h, n, theta,
                                                               chi, xi, A_pi_here, A_pi_rs_traced_here,
                                                               A_pi_rs_traced_pp_here, piunitcell)
           + Emin(np.array([kx, ky, kz - step]), lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, piunitcell)) / step ** 2
    return np.array([[fxx, fxy, fxz],[fxy, fyy, fyz],[fxz, fyz, fzz]])

def findminLam(M, K, tol, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
               BZres, kappa, equi_class_field, equi_class_flux, gen_equi_class_field, gen_equi_class_flux):
    if Jpm==0 and Jpmpm == 0 and h == 0:
        return 1/(2*kappa**2), np.array([0,0,0]).reshape((1,3))
    warnings.filterwarnings("error")
    E, V = np.linalg.eigh(M)
    E = E[:,0]
    Em = E.min()
    dex = np.where(np.abs(E-Em)<5e-16)
    Know = K[dex]
    Know = symmetry_equivalence(Know, equi_class_flux)
    Know = symmetry_equivalence(Know, equi_class_field)

    if Know.shape == (3,):
        Know = Know.reshape(1,3)

    if len(Know) >= 8:
        Know = Know[0:8]
    step = 1
    Enow = Em*np.ones(len(Know))
    for i in range(len(Know)):
        stuff = True
        init = True
        while stuff:
            if not init:
                gradlen = gradient(Know[i], np.zeros(2), Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here,
                                   A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - gradient(Klast, np.zeros(2), Jpm,
                                                                                           Jpmpm, h, n, theta, chi, xi,
                                                                                           A_pi_here,
                                                                                           A_pi_rs_traced_here,
                                                                                           A_pi_rs_traced_pp_here)
                try:
                    step = abs(np.dot(Know[i] - Klast, gradlen)) / np.linalg.norm(gradlen) ** 2
                except:
                    step = 0

            Klast = np.copy(Know[i])
            Know[i] = Know[i] - step * gradient(Know[i], np.zeros(2), Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here,
                                                A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
            Elast = np.copy(Enow[i])
            Enow[i] = Emin(Know[i], np.zeros(2), Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                           A_pi_rs_traced_pp_here, piunitcell)
            init = False
            if abs(Elast-Enow[i])<1e-14:
                stuff = False
    warnings.resetwarnings()

    Enowm = Enow.min()
    dex = np.where(abs(Enow-Enowm)<5e-16)
    Know = Know[dex]
    if Know.shape == (3,):
        Know = Know.reshape(1,3)
    KnowF = gen_equi_class_field(Know)
    KnowF = gen_equi_class_flux(KnowF)

    KnowF = np.unique(np.mod(KnowF, 1),axis=0)
    Know = np.unique(np.mod(Know, 1),axis=0)
    if KnowF.shape == (3,):
        KnowF = KnowF.reshape(1,3)
    return -Enowm, KnowF, Know

#endregion

#region find minlam scipy
def findminLam_scipy(M, K, tol, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitcell, BZres, kappa):
    if Jpm==0 and Jpmpm == 0 and h == 0:
        return 1/(2*kappa**2), np.array([0,0,0]).reshape((1,3))

    E, V = np.linalg.eigh(M)
    E = E[:,0]
    Em = E.min()
    dex = np.where(np.abs(E-Em)<3e-16)
    Know = K[dex]
    Know = np.unique(np.mod(Know, 1), axis=0)
    if Know.shape == (3,):
        Know = Know.reshape(1,3)

    if len(Know) >= minLamK:
        Know = Know[0:minLamK]

    Enow = np.zeros(len(Know))
    for i in range(len(Know)):
        res = minimize(Emin, x0=Know[i], args=(np.zeros(2), Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here,
                                               A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitcell),
                       method='Nelder-Mead', bounds=((Know[i,0]-1/BZres, Know[i,0]+1/BZres), (Know[i,1]-1/BZres,Know[i,1]+1/BZres), (Know[i,2]-1/BZres,Know[i,2]+1/BZres)))
        Know[i] = np.array(res.x)
        Enow[i] = res.fun
    Enowm = Enow.min()
    dex = np.where(abs(Enow-Enowm)<3e-16)
    Know = Know[dex]
    Know = np.unique(np.mod(Know, 1),axis=0)
    if Know.shape == (3,):
        Know = Know.reshape(1,3)
    return -Enowm, Know
def findlambda_pi(kappa, tol, lamM, Jzz, weights, E):
    warnings.filterwarnings("error")
    lamMin = np.copy(lamM)
    lamMax = 2*lamMin
    lams = lamMax
    diverge = False
    while True:
        lamlast = np.copy(lams)
        lams = (lamMax+lamMin)/2
        try:
            rhoguess = rho_true(weights, E, lams, Jzz)
            error = rhoguess-kappa
            if error > 0:
                lamMin = lams
            else:
                lamMax = lams
            if ((np.absolute(rhoguess - kappa) <= tol).all()):
                break
        except:
            lamMin = lams
        if (abs(lamMin - lamMax) < 1e-15).all():
            diverge=True
            break
    warnings.resetwarnings()
    return lams, diverge

#endregion

#region Mean field calculation

def chi_integrand(k, E, V, Jzz, unitcell):
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jlk->ijl', k, NNminus)
    ffactB = np.exp(-1j * ffact)
    size = int(E.shape[1]/4)
    B = contract('iab, ijl,jka, lkb->ikjl', green[:, 3*size:4*size, size:2*size], ffactB, unitcell, unitcell)/size
    ffactA = np.exp(1j * ffact)
    A = contract('iab, ijl,jka, lkb->ikjl', green[:, 2*size:3*size, 0:size], ffactA, unitcell, unitcell)/size
    return A, B

def chiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, chi_field, nS):
    A, B = chi_integrand(pts, E, V, Jzz, unitcellGraph)
    A = contract('ikjl,i->kjl', A, weights)
    B = contract('ikjl,i->kjl', B, weights)
    M1 = chi_field(n, n1, n2, unitcellCoord, B[0], A[0], nS)
    return M1

def xi_integrand(k, E, V, Jzz, unitcellGraph):
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jk->ij', k, NN)
    ffactA = np.exp(1j * ffact)
    size = int(E.shape[1]/4)
    A = contract('ika, ij,jka->kij', green[:, 0:size, size:2*size], ffactA, unitcellGraph)/size
    return A
def xiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, xi_field, nS):
    M = integrate_fixed(xi_integrand, weights, pts, E, V, Jzz, unitcellGraph)
    M1 = xi_field(n, n1, n2, unitcellCoord, M[0], nS)
    return M1

def calmeanfield(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, xi_field, chi_field, nS):
    chi = chiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, chi_field, nS)
    # chi = np.zeros((len(unitcellCoord),4,4))
    return chi, xiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, xi_field, nS)

# endregion

#region graphing BZ

def dispersion_pi(lams, k, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, unitcell):
    temp = np.sqrt(2 * Jzz * E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                  A_pi_rs_traced_pp_here, unitcell)[0])
    return temp


def calDispersion(lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,axes,
                  unitcell=piunitcell):
    dGammaX = dispersion_pi(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                            A_pi_rs_traced_pp_here, unitcell)
    dXW = dispersion_pi(lams, XW, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                        A_pi_rs_traced_pp_here, unitcell)
    dWK = dispersion_pi(lams, WK, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                        A_pi_rs_traced_pp_here, unitcell)
    dKGamma = dispersion_pi(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                            A_pi_rs_traced_pp_here, unitcell)
    dGammaL = dispersion_pi(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                            A_pi_rs_traced_pp_here, unitcell)
    dLU = dispersion_pi(lams, LU, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                        A_pi_rs_traced_pp_here, unitcell)
    dUW1 = dispersion_pi(lams, UW1, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                        A_pi_rs_traced_pp_here, unitcell)
    dW1X1 = dispersion_pi(lams, W1X1, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                        A_pi_rs_traced_pp_here, unitcell)
    dX1Gamma = dispersion_pi(lams, X1Gamma, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                        A_pi_rs_traced_pp_here, unitcell)

    for i in range(dGammaX.shape[1]):
        axes.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX[:, i], 'b')
        axes.plot(np.linspace(gX, gW, len(dXW)), dXW[:, i], 'b')
        axes.plot(np.linspace(gW, gK, len(dWK)), dWK[:, i], 'b')
        axes.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma[:, i], 'b')
        axes.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL[:, i], 'b')
        axes.plot(np.linspace(gL, gU, len(dLU)), dLU[:, i], 'b')
        axes.plot(np.linspace(gU, gW1, len(dUW1)), dUW1[:, i], 'b')
        axes.plot(np.linspace(gW1, gX1, len(dW1X1)), dW1X1[:, i], 'b')
        axes.plot(np.linspace(gX1, gGamma3, len(dX1Gamma)), dX1Gamma[:, i], 'b')


    axes.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
    axes.axvline(x=gX, color='b', label='axvline - full height', linestyle='dashed')
    axes.axvline(x=gW, color='b', label='axvline - full height', linestyle='dashed')
    axes.axvline(x=gK, color='b', label='axvline - full height', linestyle='dashed')
    axes.axvline(x=gGamma2, color='b', label='axvline - full height', linestyle='dashed')
    axes.axvline(x=gL, color='b', label='axvline - full height', linestyle='dashed')
    axes.axvline(x=gU, color='b', label='axvline - full height', linestyle='dashed')
    axes.axvline(x=gW1, color='b', label='axvline - full height', linestyle='dashed')
    axes.axvline(x=gX1, color='b', label='axvline - full height', linestyle='dashed')
    axes.axvline(x=gGamma3, color='b', label='axvline - full height', linestyle='dashed')

    xlabpos = [gGamma1, gX, gW, gK, gGamma2, gL, gU, gW1, gX1, gGamma3]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W^\prime$', r'$X^\prime$', r'$\Gamma$']
    axes.set_xticks(xlabpos, labels)
    axes.set_xlim([0,gGamma3])
    cluster = np.concatenate((dGammaX, dXW, dWK, dKGamma, dGammaL, dLU, dUW1, dW1X1, dX1Gamma))
    axes.set_ylim([np.min(cluster)*0.5, np.max(cluster)*1.2])
#endregion

#region lower and upper edges
# @nb.njit(parallel=True, cache=True)
def minCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
           unitcell):
    temp = np.zeros(len(q))
    mins = dispersion_pi(lams, K, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                         A_pi_rs_traced_pp_here, unitcell)[:, 0]
    for i in range(len(q)):
        temp[i] = np.min(
            dispersion_pi(lams, K - q[i], Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                          A_pi_rs_traced_pp_here, unitcell)[:, 0]
            + mins)
    return temp


# @nb.njit(parallel=True, cache=True)
def maxCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
           unitcell):
    temp = np.zeros(len(q))
    maxs = dispersion_pi(lams, K, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                         A_pi_rs_traced_pp_here, unitcell)[:, -1]
    for i in range(len(q)):
        temp[i] = np.max(
            dispersion_pi(lams, K - q[i], Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                          A_pi_rs_traced_pp_here, unitcell)[:, -1]
            + maxs)
    return temp


def minMaxCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
              unitcell):
    temp = np.zeros((len(q), 2))
    Ek = dispersion_pi(lams, K, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                       A_pi_rs_traced_pp_here, unitcell)
    mins = Ek[:,0]
    maxs = Ek[:,-1]
    for i in range(len(q)):
        tt = dispersion_pi(lams, K - q[i], Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                           A_pi_rs_traced_pp_here, unitcell)
        temp[i, 0] = np.min(tt[:, 0] + mins)
        temp[i, 1] = np.max(tt[:, -1] + maxs)
    return temp

def DSSF_E_Low(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
               A_pi_rs_traced_pp_here):
    Eq = np.sqrt(2 * Jzz * E_pi(K, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here)[0])
    Ek = np.sqrt(2 * Jzz * E_pi(K - q, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here)[0])
    return min(Eq[:,0]+Ek[:,0])

def DSSF_E_High(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                A_pi_rs_traced_pp_here):
    Eq = np.sqrt(2 * Jzz * E_pi(K, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here)[0])
    Ek = np.sqrt(2 * Jzz * E_pi(K - q, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here)[0])
    return max(Eq[:,-1]+Ek[:,-1])

def DSSF_E_DOMAIN(lams, qmin, qmax, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here):
    return DSSF_E_Low(lams, qmin, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                      A_pi_rs_traced_pp_here) \
        , DSSF_E_High(lams, qmax, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                      A_pi_rs_traced_pp_here)


def loweredge(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
              unitcell ,ax):
    dGammaX = minCal(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dXW = minCal(lams, XW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dWK = minCal(lams, WK, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dLU = minCal(lams, LU, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dUW1 = minCal(lams, UW1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dW1X1 = minCal(lams, W1X1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dX1Gamma = minCal(lams, X1Gamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)

    ax.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, 'w',zorder=8)
    ax.plot(np.linspace(gX, gW, len(dXW)), dXW, 'w',zorder=8)
    ax.plot(np.linspace(gW, gK, len(dWK)), dWK, 'w',zorder=8)
    ax.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, 'w',zorder=8)
    ax.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, 'w',zorder=8)
    ax.plot(np.linspace(gL, gU, len(dLU)), dLU, 'w',zorder=8)
    ax.plot(np.linspace(gU, gW1, len(dUW1)), dUW1, 'w',zorder=8)
    ax.plot(np.linspace(gW1, gX1, len(dW1X1)), dW1X1, 'w',zorder=8)
    ax.plot(np.linspace(gX1, gGamma3, len(dX1Gamma)), dX1Gamma, 'w',zorder=8)

    ax.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gW, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gW1, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gX1, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma3, color='w', label='axvline - full height', linestyle='dashed')

    xlabpos = [gGamma1, gX, gW, gK, gGamma2, gL, gU, gW1*1.02, gX1, gGamma3]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W^\prime$', r'$X^\prime$', r'$\Gamma$']
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0,gGamma3])

    return np.concatenate((dGammaX, dXW, dWK, dKGamma, dGammaL, dLU, dUW1, dW1X1, dX1Gamma))

def upperedge(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
              unitcell ,ax):
    dGammaX = maxCal(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dXW = maxCal(lams, XW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dWK = maxCal(lams, WK, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dKGamma = maxCal(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dGammaL = maxCal(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dLU = maxCal(lams, LU, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dUW1 = maxCal(lams, UW1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dW1X1 = maxCal(lams, W1X1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dX1Gamma = maxCal(lams, X1Gamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)


    ax.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, 'w',zorder=8)
    ax.plot(np.linspace(gX, gW, len(dXW)), dXW, 'w',zorder=8)
    ax.plot(np.linspace(gW, gK, len(dWK)), dWK, 'w',zorder=8)
    ax.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, 'w',zorder=8)
    ax.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, 'w',zorder=8)
    ax.plot(np.linspace(gL, gU, len(dLU)), dLU, 'w',zorder=8)
    ax.plot(np.linspace(gU, gW1, len(dUW1)), dUW1, 'w',zorder=8)
    ax.plot(np.linspace(gW1, gX1, len(dW1X1)), dW1X1, 'w',zorder=8)
    ax.plot(np.linspace(gX1, gGamma3, len(dX1Gamma)), dX1Gamma, 'w',zorder=8)

    ax.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gW, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gW1, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gX1, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=gGamma3, color='w', label='axvline - full height', linestyle='dashed')

    xlabpos = [gGamma1, gX, gW, gK, gGamma2, gL, gU, gW1*1.02, gX1, gGamma3]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W^\prime$', r'$X^\prime$', r'$\Gamma$']
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0,gGamma3])

    return np.concatenate((dGammaX, dXW, dWK, dKGamma, dGammaL, dLU, dUW1, dW1X1, dX1Gamma))
def loweredge_data(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
              unitcell):
    dGammaX = minCal(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dXW = minCal(lams, XW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dWK = minCal(lams, WK, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dLU = minCal(lams, LU, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dUW1 = minCal(lams, UW1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dW1X1 = minCal(lams, W1X1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dX1Gamma = minCal(lams, X1Gamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)

    return np.concatenate((dGammaX, dXW, dWK, dKGamma, dGammaL, dLU, dUW1, dW1X1, dX1Gamma))

def upperedge_data(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
              unitcell):
    dGammaX = maxCal(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dXW = maxCal(lams, XW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dWK = maxCal(lams, WK, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dKGamma = maxCal(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dGammaL = maxCal(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, unitcell)
    dLU = maxCal(lams, LU, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dUW1 = maxCal(lams, UW1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dW1X1 = maxCal(lams, W1X1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)
    dX1Gamma = maxCal(lams, X1Gamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)

    return np.concatenate((dGammaX, dXW, dWK, dKGamma, dGammaL, dLU, dUW1, dW1X1, dX1Gamma))


#endregion

#region greens function and energetics
def green_pi(E, V, Jzz):
    green = contract('ilk, ijk, ik->ijl', V, np.conj(V), Jzz / E)
    return green

def green_pi_branch(E, V, Jzz):
    green = contract('ilk, ijk, ik->ikjl', V, np.conj(V), Jzz / E)
    return green

#endregion

#region miscellaneous
def gap(M, lams):
    # temp = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    temp = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
    E, V = np.linalg.eigh(temp)
    # E = np.sqrt(E)
    temp = np.amin(E)
    return temp


def EMAX(M, lams):
    # temp = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    temp = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
    E, V = np.linalg.eigh(temp)
    temp = np.amax(E)
    return temp

def graphing_M_setup(flux, n):
    if (flux == np.zeros(4)).all():
        unitCellgraph = np.array([[[1]],[[1]],[[1]],[[1]]])
        A_pi_here = np.array([[0,0,0,0]])
        unitcellCoord = np.array([[0,0,0]])
    elif (flux == np.pi*np.ones(4)).all():
        unitCellgraph = piunitcell
        if (n==h111).all():
            A_pi_here = A_pi
        elif (n==h001).all():
            A_pi_here = np.array([[0, 0, np.pi, 0],
                             [0, np.pi, np.pi, 0],
                             [0, np.pi, 0, 0],
                             [0, 0, 0, 0]])
        else:
            A_pi_here = np.array([[0, 0, 0, np.pi],
                             [0, np.pi, 0, np.pi],
                             [0, np.pi, np.pi, np.pi],
                             [0, 0, np.pi, np.pi]])
        unitcellCoord = np.array([[0, 0, 0],[0,1,0],[0,0,1],[0,1,1]])
    elif (flux == np.array([np.pi,np.pi,0,0])).all():
        unitCellgraph = np.array([[[1,0],
                                    [0,1]],
                                    [[1,0],
                                    [0,1]],
                                    [[0,1],
                                    [1,0]],
                                    [[1,0],
                                    [0,1]]
                            ])
        A_pi_here = np.array([[0,0,0,0],
                                [0,np.pi,0,0]])
        unitcellCoord = np.array([[0, 0, 0],[0,1,0]])
    elif (flux == np.array([0, 0, np.pi, np.pi])).all():
        unitCellgraph = np.array([[[1,0],
                                    [0,1]],
                                    [[1,0],
                                    [0,1]],
                                    [[1,0],
                                    [0,1]],
                                    [[0,1],
                                    [1,0]]
                            ])
        A_pi_here = np.array([[0,0,0,0],
                                [0,np.pi,np.pi,0]])
        unitcellCoord = np.array([[0, 0, 0],[0,0,1]])

    # elif (flux == np.array([np.pi,0,0, np.pi])).all():
    #     unitCellgraph = np.array([[[1,0],
    #                                 [0,1]],
    #                                 [[1,0],
    #                                 [0,1]],
    #                                 [[0,1],
    #                                 [1,0]],
    #                                 [[0,1],
    #                                 [1,0]]
    #                         ])
    #     A_pi_here = np.array([[0,0,0,0],
    #                             [0,np.pi,0,0]])
    #     unitcellCoord = np.array([[0, 0, 0],[0,1,1]])
    #
    # elif (flux == np.array([0, np.pi, np.pi, 0])).all():
    #     unitCellgraph = np.array([[[1,0],
    #                                 [0,1]],
    #                                 [[1,0],
    #                                 [0,1]],
    #                                 [[1,0],
    #                                 [0,1]],
    #                                 [[0,1],
    #                                 [1,0]]
    #                         ])
    #     A_pi_here = np.array([[0,0,0,0],
    #                             [0,0,np.pi,0]])
    #     unitcellCoord = np.array([[0, 0, 0],[0,1,0]])
    return unitCellgraph, A_pi_here, unitcellCoord

def xi_wo_field(n, n1, n2, unitcellcoord, xi0, *args):
    #in the case of 110, three xi mf: xi0, xi1, xi3
    mult = np.zeros((len(unitcellcoord),4),dtype=np.complex128)
    nS, = args
    for i in range(len(unitcellcoord)):
        mult[i] = np.array([xi0[0], xi0[0]*np.exp(1j*np.pi*(nS+n1*(unitcellcoord[i,1]+unitcellcoord[i,2]))), xi0[0]*np.exp(1j*np.pi*(nS+n1*unitcellcoord[i,2])), xi0[0]*np.exp(1j*np.pi*nS)])
    return mult

def chi_wo_field(n, n1, n2, unitcellCoord, chi0, chi0A, *args):

    mult = np.zeros((2, len(unitcellCoord),4, 4),dtype=np.complex128)

    for i in range(len(unitcellCoord)):
        r2 = unitcellCoord[i,1]
        r3 = unitcellCoord[i,2]

        chi00 = chi0[0,0]

        chi01 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
        chi02 = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))
        chi03 = chi0[0,3]
        chi12 = chi0[0,3]*np.exp(1j*np.pi*(n1*(r2+1)))
        chi13 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3+n1))
        chi23 = chi0[0,1]*np.exp(1j*np.pi*n1*(r3+1))

        chi00A = chi0[0,0]
        chi01A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
        chi02A = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))
        chi03A = chi0[0,3]
        chi12A = chi0[0,3]*np.exp(1j*np.pi*(n1*(r2+1)))
        chi13A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3+n1))
        chi23A = chi0[0,1]*np.exp(1j*np.pi*n1*(r3+1))

        mult[1, i] = np.array([[chi00, chi01, chi02, chi03],
                               [chi01, chi00, chi12, chi13],
                               [chi02, chi12, chi00, chi23],
                               [chi03, chi13, chi23, chi00]])
        mult[0, i] = np.array([[chi00A, chi01A, chi02A, chi03A],
                               [chi01A, chi00A, chi12A, chi13A],
                               [chi02A, chi12A, chi00A, chi23A],
                               [chi03A, chi13A, chi23A, chi00A]])
    return mult
def xi_w_field_Octu(n, n1, n2, unitcellcoord, xi0):
    #in the case of 110, three xi mf: xi0, xi1, xi3
    mult = np.zeros((len(unitcellcoord),4),dtype=np.complex128)
    for i in range(len(unitcellcoord)):
        if (n==h110).all():
            mult[i] = np.array([xi0[0], xi0[1]*np.exp(1j*(n1*np.pi*unitcellcoord[i,1]+n2*np.pi*unitcellcoord[i,2])), xi0[2]*np.exp(1j*(n2*np.pi*unitcellcoord[i,2])), xi0[0]])
        elif (n==h111).all():
            mult[i] = np.array([xi0[0], xi0[1]*np.exp(1j*(n1*np.pi*unitcellcoord[i,1]+n1*np.pi*unitcellcoord[i,2])), xi0[1]*np.exp(1j*(n1*np.pi*unitcellcoord[i,2])), xi0[1]])
        else:
            mult[i] = np.array([xi0[0], xi0[0]*np.exp(1j*(n1*np.pi*unitcellcoord[i,1]+n1*np.pi*unitcellcoord[i,2])), xi0[0]*np.exp(1j*(n1*np.pi*unitcellcoord[i,2])), xi0[0]])
    return mult

def chi_w_field_Octu(n, n1, n2, unitcellCoord, chi0, chi0A, *args):

    mult = np.zeros((2, len(unitcellCoord),4, 4),dtype=np.complex128)

    for i in range(len(unitcellCoord)):
        r2 = unitcellCoord[i,1]
        r3 = unitcellCoord[i,2]

        chi00 = chi0[0,0]

        if (n==h110).all():

            psiIT1, psiIT2, psiI = args

            chi01 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n2*r3))
            chi02 = chi0[0,2]*np.exp(1j*np.pi*(n2*r3))
            chi03 = chi0[0,3]
            chi12 = chi0[1,2]*np.exp(1j*np.pi*(n1*r2))
            chi13 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n2*(r3+1)))
            chi23 = chi0[1,2]*np.exp(1j*np.pi*n2*r3)

            chi00A = chi0[0,0]/psiI
            chi01A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n2*r3))*psiIT1/psiI
            chi02A = chi0[0,2]*np.exp(1j*np.pi*(n2*r3))*psiIT2/psiI
            chi03A = chi0[0,3]/psiI
            chi12A = chi0[1,2]*np.exp(1j*np.pi*(n1*r2))*psiIT2/psiI/psiIT1
            chi13A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n2*(r3+1)))/psiI/psiIT1
            chi23A = chi0[1,2]*np.exp(1j*np.pi*n2*r3)/psiI/psiIT2

        elif (n==h111).all():
            psiC6, = args

            chi01 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi02 = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))*psiC6**(-4/3)
            chi03 = chi0[0,1]*psiC6**(-2/3)
            chi12 = chi0[2,3]*np.exp(1j*np.pi*(n1*r2))*psiC6**(-2/3)
            chi13 = chi0[2,3]*np.exp(1j*np.pi*(n1*r2+n2*r3))*psiC6**(-4/3)
            chi23 = chi0[2,3]*np.exp(1j*np.pi*n1*r3)

            chi00A = chi0[0,0]/psiC6
            chi01A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))/psiC6
            chi02A = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))*psiC6**(-7/3)
            chi03A = chi0[0,1]*psiC6**(-5/3)
            chi12A = chi0[2,3]*np.exp(1j*np.pi*(n1*r2))*psiC6**(-5/3)
            chi13A = chi0[2,3]*np.exp(1j*np.pi*(n1*r2+n2*r3))*psiC6**(-7/3)
            chi23A = chi0[2,3]*np.exp(1j*np.pi*n1*r3)/psiC6


        else:
            chi01 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi02 = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))
            chi03 = chi0[0,3]
            chi12 = chi0[0,3]*np.exp(1j*np.pi*(n1*r2))
            chi13 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi23 = chi0[0,1]*np.exp(1j*np.pi*n1*r3)

            chi00A = chi0A[0,0]
            chi01A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi02A = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))
            chi03A = chi0[0,3]
            chi12A = chi0[0,3]*np.exp(1j*np.pi*(n1*r2))
            chi13A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi23A = chi0[0,1]*np.exp(1j*np.pi*n1*r3)

        mult[1, i] = np.array([[chi00, chi01, chi02, chi03],
                               [chi01, chi00, chi12, chi13],
                               [chi02, chi12, chi00, chi23],
                               [chi03, chi13, chi23, chi00]])
        mult[0, i] = np.array([[chi00A, chi01A, chi02A, chi03A],
                               [chi01A, chi00A, chi12A, chi13A],
                               [chi02A, chi12A, chi00A, chi23A],
                               [chi03A, chi13A, chi23A, chi00A]])
    return mult

def xi_w_field_Diu(n, n1, n2, unitcellcoord, xi0):
    #in the case of 110, three xi mf: xi0, xi1, xi3
    mult = np.zeros((len(unitcellcoord),4),dtype=np.complex128)
    for i in range(len(unitcellcoord)):
        if (n==h110).all():
            mult[i] = np.array([xi0[0], xi0[1]*np.exp(1j*(n1*np.pi*unitcellcoord[i,1]+n2*np.pi*unitcellcoord[i,2])), xi0[2]*np.exp(1j*(n2*np.pi*unitcellcoord[i,2])), xi0[0]])
        elif (n==h111).all():
            mult[i] = np.array([xi0[0], xi0[1]*np.exp(1j*(n1*np.pi*unitcellcoord[i,1]+n1*np.pi*unitcellcoord[i,2])), xi0[1]*np.exp(1j*(n1*np.pi*unitcellcoord[i,2])), xi0[1]])
        else:
            mult[i] = np.array([xi0[0], xi0[0]*np.exp(1j*(n1*np.pi*unitcellcoord[i,1]+n1*np.pi*unitcellcoord[i,2])), xi0[0]*np.exp(1j*(n1*np.pi*unitcellcoord[i,2])), xi0[0]])
    return mult

def chi_w_field_Diu(n, n1, n2, unitcellCoord, chi0, chi0A, *args):

    mult = np.zeros((2, len(unitcellCoord),4, 4),dtype=np.complex128)

    for i in range(len(unitcellCoord)):
        r2 = unitcellCoord[i,1]
        r3 = unitcellCoord[i,2]

        chi00 = chi0[0,0]
        if (n==h110).all():
            psisigmaT1, psisigmaT2, psiI, nI = args

            chi01 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n2*r3))
            chi02 = chi0[0,2]*np.exp(1j*np.pi*(n2*r3))
            chi03 = chi0[0,3]
            chi12 = chi0[1,2]*np.exp(1j*np.pi*(n1*r2))
            chi13 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n2*(r3+1)))*psisigmaT1
            chi23 = chi0[1,2]*np.exp(1j*np.pi*n2*r3)*psisigmaT2*psisigmaT1

            chi00A = chi0[0,0]/psiI
            chi01A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n2*r3))/psiI*(-1)**nI
            chi02A = chi0[0,2]*np.exp(1j*np.pi*(n2*r3))/psiI*(-1)**nI
            chi03A = chi0[0,3]/psiI
            chi12A = chi0[1,2]*np.exp(1j*np.pi*(n1*r2))/psiI
            chi13A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n2*(r3+1)))/psiI*(-1)**nI*psisigmaT1
            chi23A = chi0[1,2]*np.exp(1j*np.pi*n2*r3)/psiI*psisigmaT2*(-1)**nI*psisigmaT1

        elif (n==h111).all():
            psiC6, = args

            chi01 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi02 = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))*psiC6**(-4/3)
            chi03 = chi0[0,1]*psiC6**(-2/3)
            chi12 = chi0[2,3]*np.exp(1j*np.pi*(n1*r2))*psiC6**(-2/3)
            chi13 = chi0[2,3]*np.exp(1j*np.pi*(n1*r2+n2*r3))*psiC6**(-4/3)
            chi23 = chi0[2,3]*np.exp(1j*np.pi*n1*r3)

            chi00A = chi0[0,0]/psiC6
            chi01A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))/psiC6
            chi02A = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))*psiC6**(-7/3)
            chi03A = chi0[0,1]*psiC6**(-5/3)
            chi12A = chi0[2,3]*np.exp(1j*np.pi*(n1*r2))*psiC6**(-5/3)
            chi13A = chi0[2,3]*np.exp(1j*np.pi*(n1*r2+n2*r3))*psiC6**(-7/3)
            chi23A = chi0[2,3]*np.exp(1j*np.pi*n1*r3)/psiC6


        else:

            psiS4, psiI = args

            chi01 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi02 = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))/psiS4
            chi03 = chi0[0,3]
            chi12 = chi0[0,3]*np.exp(1j*np.pi*(n1*r2))/psiS4
            chi13 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))/psiS4
            chi23 = chi0[0,1]*np.exp(1j*np.pi*n1*r3)/psiS4**2

            chi00A = chi0[0,0]
            chi01A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))/psiI
            chi02A = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))/psiS4/psiI
            chi03A = chi0[0,3]/psiI
            chi12A = chi0[0,3]*np.exp(1j*np.pi*(n1*r2))/psiS4/psiI
            chi13A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))/psiS4/psiI
            chi23A = chi0[0,1]*np.exp(1j*np.pi*n1*r3)/psiS4**2/psiI

        mult[1, i] = np.array([[chi00, chi01, chi02, chi03],
                               [chi01, chi00, chi12, chi13],
                               [chi02, chi12, chi00, chi23],
                               [chi03, chi13, chi23, chi00]])
        mult[0, i] = np.array([[chi00A, chi01A, chi02A, chi03A],
                               [chi01A, chi00A, chi12A, chi13A],
                               [chi02A, chi12A, chi00A, chi23A],
                               [chi03A, chi13A, chi23A, chi00A]])
    return mult

# def xiwithfieldOctu(n, n1, n2, r2, r3):
#     if (n==h110).all():
#         mult = np.array([[1, ],
#                          ])
#         same = np.array([[0, 1, 2, 3],
#                          [1, 0, 4, 1],
#                          [2, 4, 0, 4],
#                          [3, 1, 4, 0]])
#     elif (n==h110).all():
#         same = np.array([[0, 1, 2, 3],
#                          [1, 0, 4, 1],
#                          [2, 4, 0, 4],
#                          [3, 1, 4, 0]])
#     else:
#         same = np.array([[0, 1, 2, 3],
#                          [1, 0, 4, 1],
#                          [2, 4, 0, 4],
#                          [3, 1, 4, 0]])
#endregion
class piFluxSolver:
    def __init__(self, Jxx, Jyy, Jzz, theta=0, h=0, n=h110, kappa=2, lam=2, BZres=20, graphres=20,
                 ns=1, tol=1e-10, flux=np.zeros(4), intmethod=gauss_quadrature_3D_pts, gzz=2.24, Breal=False, nS=0):
        self.intmethod = intmethod
        J = np.array([Jxx, Jyy, Jzz])
        print("Instance Created with parameters " + str(J) + " with flux " + str(flux))
        a = np.argmax(J)
        xx = np.mod(a-2,3)
        yy = np.mod(a-1,3)
        self.dominant = a
        self.Jzz = J[a]
        self.Jpm = -(J[xx] + J[yy]) / 4
        self.Jpmpm = (J[xx] - J[yy]) / 4
        self.theta = theta
        self.kappa = kappa
        self.tol = tol
        self.lams = np.array([lam, lam], dtype=np.double)
        self.nS = nS

        if h == 0:
            self.xi_field = xi_wo_field
            self.chi_field = chi_wo_field
        elif a == 1 or a == 2:
            self.xi_field = xi_w_field_Octu
            self.chi_field = chi_w_field_Octu
        else:
            self.xi_field = xi_w_field_Diu
            self.chi_field = chi_w_field_Diu

        if Breal:
            self.h = 5.7883818060*10**(-2)*h*gzz
        else:
            self.h = h
        if a == 0:
            self.h = -1j*self.h
        self.n = n
        self.flux = flux
        self.A_pi_here, self.n1, self.n2, self.equi_class_field, self.equi_class_flux, self.gen_equi_class_field, self.gen_equi_class_flux = determineEquivalence(n, flux)
        self.pts, self.weights = self.intmethod(0, 1, 0, 1, 0, 1, BZres)

        self.minLams = np.zeros(2, dtype=np.double)
        self.BZres = BZres
        self.graphres = graphres

        self.toignore = np.array([],dtype=int)
        self.q = np.nan
        self.qmin = np.empty(3)
        self.qmin[:] = np.nan
        self.qminB = np.copy(self.qmin)
        self.condensed = False
        self.delta = np.zeros(16)
        self.rhos = np.zeros(16)

        # self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.A_pi_rs_rsp_here, self.A_pi_rs_rsp_pp_here = gen_gauge_configurations(self.A_pi_here)
        # self.unitCellgraph = piunitcell
        self.unitCellgraph, self.A_pi_here, self.unitcellCoord = graphing_M_setup(self.flux, self.n)
        self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.A_pi_rs_rsp_here, self.A_pi_rs_rsp_pp_here = gen_gauge_configurations(
            self.A_pi_here)
        self.xi = self.xi_field(n, self.n1, self.n2, self.unitcellCoord, 0.05*np.ones(4), self.nS)
        self.chi = self.chi_field(n, self.n1, self.n2, self.unitcellCoord, 0.02*np.ones((4,4)), 0.05*np.ones((4,4)), self.nS)
        self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                       self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)
        self.E, self.V = np.linalg.eigh(self.MF)

    def findLambda(self, a=True):
        if a:
            return findlambda_pi(self.kappa, self.tol,self.minLams, self.Jzz, self.weights, self.E)
        else:
            return findlambda_pi(self.kappa, self.tol,np.abs(np.min(self.E))*0.8*np.ones(2), self.Jzz, self.weights, self.E)

    def findLambda_unconstrained(self):
        return findlambda_pi(self.kappa,self.tol, np.zeros(2), self.Jzz, self.weights, self.E)


    def findminLam(self):
        searchGrid=34
        B = genBZ(searchGrid)
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux, self.n)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(A_pi_here)
        M = M_pi(B, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, A_pi_here,
                 A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)
        minLams, self.qmin = findminLam_scipy(M, B, self.tol, self.Jpm, self.Jpmpm, self.h, self.n,
                                        self.theta, self.chi, self.xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph,
                                        searchGrid, self.kappa)
        self.qminB = contract('ij,jk->ik', self.qmin, BasisBZA)
        self.minLams = np.ones(2) * minLams
        return minLams

    def rho(self,lam):
        A = np.delete(self.weights, self.toignore)
        B = np.delete(self.E, self.toignore, axis=0)
        return rho_true(A, B, lam,self.Jzz)
    def rho_site(self,lam):
        A = np.delete(self.weights, self.toignore)
        B = np.delete(self.E, self.toignore, axis=0)
        C = np.delete(self.V, self.toignore, axis=0)
        return rho_true_site(A, B,C, lam,self.Jzz)
    def calmeanfield(self):
        E, V = self.LV_zero(self.pts, self.lams)
        E = np.sqrt(2*self.Jzz*E)
        chi, xi = calmeanfield(E, V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.chi_field, self.nS)
        return chi, xi

    def solvexifield(self):
        print(self.lams, self.minLams)
        E, V = self.LV_zero(self.pts, self.lams)
        E = np.sqrt(2*self.Jzz*E)
        xi = xiCal(E, V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.nS)
        return xi

    def solvechifield(self):
        E, V = self.LV_zero(self.pts, self.lams)
        E = np.sqrt(2*self.Jzz*E)
        chi = chiCal(E, V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.nS)
        return chi

    def updateMF(self):
        self.M = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                 self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)

    def xiSubrountine(self, tol, GS, pcon=False):
        if pcon:
            count = 5
        else:
            count = 0
        print("Begin Xi Subroutine")
        while True:
            print(self.xi)
            xilast, GSlast = np.copy(self.xi), GS
            # print("Xi Mean Field Compute")
            self.xi = self.solvexifield()
            self.updateMF()
            # print("Solve mu field")
            GS, diverge = self.solvemufield()
            if np.abs(GS) > 1e1 or diverge:
                self.xi=xilast
                print("Xi Subrountine ends. Possible Condensed Phase. Exiting Energy is: " + str(GSlast) + " Took " + str(count) + " cycles.")
                return GSlast, True
            count = count + 1
            if ((abs(GS - GSlast) < tol).all()) or count > 10:
                break
        print("Xi Subrountine ends. Exiting Energy is: "+ str(GS) + " Took " + str(count) + " cycles.")
        return GS, False

    def chiSubrountine(self, tol, GS, pcon=False):
        if pcon:
            count = 5
        else:
            count = 0
        print("Begin Chi Subroutine")
        while True:
            chilast, GSlast = np.copy(self.chi), GS
            # print("Chi Mean Field Compute")
            self.chi = self.solvechifield()
            self.updateMF()
            # print("Solve mu field")
            GS, diverge = self.solvemufield()
            if np.abs(GS) > 1e1 or diverge:
                self.chi=chilast
                print("Chi Subrountine ends. Possible Condensed Phase. Exiting Energy is: " + str(GSlast) + " Took " + str(count) + " cycles.")
                return GSlast, True
            # print(self.chi[0,0], GS)
            count = count + 1
            if ((abs(GS - GSlast) < tol).all()) or count > 10:
                break
        print("Chi Subrountine ends. Exiting Energy is: "+ str(GS) + " Took " + str(count) + " cycles.")
        return GS, False

    def solvemufield(self, a=True):
        if a:
            self.findminLam()
        self.lams, diverge = self.findLambda(a)
        return self.GS(), diverge


    def solvemeanfield(self, tol=1e-8):
        tstart = time.time()
        if self.Jpmpm == 0 and self.Jpm==0:
            self.chi = np.zeros((len(self.unitcellCoord),4,4))
            self.xi = np.zeros((4,4))
            self.condensation_check()
            self.condensed = False
        elif self.Jpmpm == 0:
            self.chi = np.zeros((len(self.unitcellCoord),4,4))
            self.xi = np.zeros((4,4))
            self.condensation_check()
        else:
            print("Initialization Routine")
            limit = 5
            self.findminLam()
            self.lams, d = self.findLambda()
            self.chi, self.xi = self.calmeanfield()
            GS, d = self.solvemufield()
            print("Initialization Routine Ends. Starting Parameters: GS="+ str(GS) + " xi0= " + str(self.xi[0]) + " chi0= " + str(self.chi[0,0]))
            count = 0
            pconxi = False
            pconChi = False
            while True:
                chilast, xilast, GSlast = np.copy(self.chi), np.copy(self.xi), np.copy(GS)
                GS, pconxi = self.xiSubrountine(tol, GS, pconxi)
                GS, pconChi = self.chiSubrountine(tol, GS, pconChi)
                print("Iteration #"+str(count))
                count = count + 1
                if pconxi or pconChi:
                    limit = 2
                if ((abs(GS-GSlast) < tol).all()) or count > limit:
                    break
            self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                           self.A_pi_here,
                           self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)
            self.E, self.V = np.linalg.eigh(self.MF)
            self.condensation_check()
            print("Finished Solving. Parameters: Jzz=" + str(self. Jzz) + "; Jpm="+str(self.Jpm)+"; Jpmpm="+str(self.Jpmpm)+"; condensed="+str(self.condensed))
        tend = time.time()
        print("This run took "+ str(tend-tstart))
        return 0

    def ifcondense(self):
        # c = np.array([])
        # if self.condensed:
        # c = np.where((self.E[0]+self.minLams[0])<=1e-6)[0]
        # self.toignore = np.array(c, dtype=int)
        # print(self.toignore)
        if self.condensed:
            self.toignore = indextoignore_tol(self.pts, self.qmin, 1e-10)
        # print(self.toignore)

    def low(self):
        E, V = np.linalg.eigh(self.MF)
        cond = np.argmin(E[:, 0])
        return self.bigB[cond], E[cond][0]

    def set_condensed(self):
        # A = self.rho(self.minLams+1e-14)
        # self.condensed = A < self.kappa
        A = -self.minLams[0] + self.lams[0]
        if A < (deltamin /(self.BZres**3)) ** 2 or not (np.abs(self.chi)<=1e-5).all():
            self.condensed = True
        else:
            self.condensed = False

    def set_delta(self):
        warnings.filterwarnings('error')
        try:
            self.rhos = np.sqrt(self.kappa - self.rho(self.minLams))*np.ones(16)
            self.delta = np.sqrt(self.Jzz/2)/self.rhos**2
        except:
            self.rhos = np.zeros(16)
            self.delta = np.zeros(16)
        warnings.resetwarnings()

    def condensation_check(self):
        self.findminLam()
        self.lams, d = self.findLambda()
        self.set_condensed()
        self.ifcondense()
        self.set_delta()


    def M_true(self, k):
        return M_pi(k, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                    self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def E_pi_mean(self, k):
        return np.mean(np.sqrt(2 * self.Jzz *
                               E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                                    self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)[0]), axis=1)

    def E_pi(self, k):
        return np.sqrt(2 * self.Jzz *
                       E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                            self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)[0])
    def E_pi_reduced(self, k):
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux, self.n)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(
            A_pi_here)
        xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        return np.sqrt(2 * self.Jzz *
                       E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi,
                            A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)[0])


    def dispersion(self, k):
        return dispersion_pi(self.lams, k, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi,
                             self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)

    def LV_zero(self, k, lam=np.zeros(2)):
        if np.any(lam == 0):
            lam = self.lams
        return E_pi(k, lam, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                    self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)

    def GS(self):
        return np.dot(self.E_pi_mean(self.pts), self.weights) - self.kappa*self.lams[0]


    def MFE(self):
        Ep = self.GS()
        return Ep

    def graph_raw(self, show):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                      self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        if show:
            plt.show()

    def graph(self, axes):
        # unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux)
        # A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(A_pi_here)
        # xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        # chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        calDispersion(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                      self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, axes, self.unitCellgraph)


    def minCal(self, K):
        xi = xiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.nS)
        chi = chiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.nS)
        return minCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, chi,
                      xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)

    def maxCal(self, K):
        xi = xiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.nS)
        chi = chiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.nS)
        return maxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, chi,
                      xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)

    def minMaxCal(self, K):
        return minMaxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, self.chi,
                         self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)

    def EMAX(self):
        return np.sqrt(2 * self.Jzz * EMAX(self.MF, self.lams))
    def TWOSPINON_GAP(self, k):
        return np.min(self.minCal(k))

    def TWOSPINON_MAX(self, k):
        return np.max(self.maxCal(k))

    def TWOSPINON_DOMAIN(self):
        searchGrid=34
        B = genBZ(searchGrid)
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux, self.n)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(A_pi_here)
        xi = xiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.nS)
        chi = chiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.nS)
        q = np.sqrt(2 * self.Jzz *
                       E_pi(B, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi,
                            A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)[0])
        mins = np.min(q[:,0])
        maxs = np.max(q[:,-1])
        return 2*mins, 2*maxs

    def graph_loweredge(self, show, ax=plt):
        xi = xiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.nS)
        chi = chiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.nS)
        min = loweredge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, chi, xi,
                  self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph, ax)
        if show:
            plt.show()
        return min

    def graph_upperedge(self, show, ax=plt):
        xi = xiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.nS)
        chi = chiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.nS)
        max = upperedge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, chi, xi,
                  self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph, ax)
        if show:
            plt.show()
        return max


    def loweredge(self):
        xi = xiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.nS)
        chi = chiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.nS)
        min = loweredge_data(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, chi, xi,
                  self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)
        return min

    def upperedge(self):
        xi = xiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.nS)
        chi = chiCal(self.E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.nS)
        max = upperedge_data(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, chi, xi,
                  self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)
        return max

    def green_pi(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi(E, V, self.Jzz)

    def green_pi_branch(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi_branch(E, V, self.Jzz), E

    def green_pi_reduced(self, k):
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux, self.n)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(
            A_pi_here)
        xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        E, V = E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi, A_pi_here,
             A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi(E, V, self.Jzz)

    def green_pi_branch_reduced(self, k):
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux, self.n)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(
            A_pi_here)
        xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        E, V = E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi, A_pi_here,
             A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi_branch(E, V, self.Jzz), E, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here, unitCellgraph

    def mag_integrand(self, k):
        E, V = E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
             self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)
        E = np.sqrt(2 * self.Jzz * E)
        green = green_pi(E, V, self.Jzz)

        ffact = contract('ik, jk->ij', k, NN)
        ffact = np.exp(1j * ffact)
        l = len(self.A_pi_here)
        zmag = contract('k,ik->i', self.n, z)
        magp = contract('ku, u, ru, urx,krx->k', ffact * (np.cos(self.theta) - 1j * np.sin(self.theta)), zmag,
                     np.exp(1j * self.A_pi_here), self.unitCellgraph, green[:,0:l,l:2*l])/(2*l)
        magn = contract('ku, u, ru, urx,krx->k', np.conj(ffact) * (np.cos(self.theta) + 1j * np.sin(self.theta)), zmag,
                     np.exp(-1j * self.A_pi_here), self.unitCellgraph, green[:,l:2*l,0:l])/(2*l)
        mag = (magp + magn)/2
        # magp = np.real(contract('ku, ru, krx, urx->rku', ffact, np.exp(1j*self.A_pi_here), green[:, 0:l, l:2*l], self.unitCellgraph))
        return mag


    def magnetization(self):
        sz = integrate(self.mag_integrand, self.pts, self.weights)
        # zmag = contract('k,ik->i', self.n, z)
        # mag = contract('ua, ru,a->r', z, sz, self.n)
        print(sz)

        # if self.condensed:
        #     mag = np.NAN
        # return np.real(mag)