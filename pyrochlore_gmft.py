import matplotlib.pyplot as plt
import warnings

import numpy as np

from misc_helper import *
from flux_stuff import *
import time
from scipy.optimize import minimize

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


def M_pi_sub_interhopping_AB(k, Jpmpm, xi, A_pi_rs_traced_pp_here, unitcell=piunitcell):
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    M1a = contract('jl, kjl, ij, kl, jkx->ikx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, xi, unitcell)
    M1b = contract('jl, kjl, il, kj, lkx->ikx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, xi, unitcell)

    M2a = contract('jl, kjl, ij, kl, jkx->ixk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(xi), unitcell)
    M2b = contract('jl, kjl, il, kj, lkx->ixk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(xi), unitcell)
    return M1a + M1b + M2a + M2b
    # return M1a + M1b


def M_pi_sub_pairing_AdAd(k, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell=piunitcell):
    d = np.ones(len(k))
    di = np.identity(unitcell.shape[1])
    M1 = contract('jl, kjl, kjl, i, km->ikm', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, chi[1], d, di)


    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    tempchi0 = chi[1, :,0,0]
    M2 = contract('jl, kjl, ijl, k, jka, lkb->iba', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0, unitcell,
                  unitcell)
    return M1 + M2
    # return M1

def M_pi_sub_pairing_BdBd(k, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell=piunitcell):
    d = np.ones(len(k))
    di = np.identity(unitcell.shape[1])
    M1 = contract('jl, kjl, kjl, i, km->ikm', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, chi[0], d, di)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * ffact)
    tempchi0 = chi[0, :,0,0]
    M2 = contract('jl, kjl, ijl, k, jka, lkb->iba', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0, unitcell,
                  unitcell)
    return M1+M2

def M_pi_sub_pairing_BB(k, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell=piunitcell):
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    tempchi0 = np.conj(chi[0, :,0,0])
    M2 = contract('jl, kjl, ijl, k, jka, lkb->iab', notrace, Jpmpm * A_pi_rs_traced_pp_here / 4, ffact, tempchi0, unitcell,
                  unitcell)
    return M2

def M_pi_fictitious_Z2_AA(k, alpha, A_pi_rs_traced_pp_here, g, unitcell=piunitcell):
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(neta(alpha)* 1j * ffact)
    M2 = contract('jl, kjl, ijl, jka, lkb->iba', notrace, g * A_pi_rs_traced_pp_here / 4, ffact, unitcell,
                  unitcell)
    return M2

def M_pi(k, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
         unitcell=piunitcell, cartesian=False):
    if not cartesian:
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


        MagAkBk = MagAkBk + M_pi_sub_interhopping_AB(k, Jpmpm, xi, A_pi_rs_traced_pp_here, unitcell)
        MagBkAk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))
        MagAnkBnk = M_pi_mag_sub_AB(-k, h, n, theta, A_pi_here, unitcell) + M_pi_sub_interhopping_AB(-k, Jpmpm, xi, A_pi_rs_traced_pp_here, unitcell)
        MagBnkAnk = np.conj(np.transpose(MagAnkBnk, (0, 2, 1)))

        MAdkAdnk = M_pi_sub_pairing_AdAd(k, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell) + np.conj(np.transpose(M_pi_fictitious_Z2_AA(k, 0, A_pi_rs_traced_pp_here, g, unitcell),(0,2,1)))
        MBdkBdnk = M_pi_sub_pairing_BdBd(k, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell) + M_pi_fictitious_Z2_AA(k, 1, A_pi_rs_traced_pp_here, g, unitcell)
        MBnkBk = np.conj(np.transpose(MBdkBdnk, (0, 2, 1)))
        MAnkAk = np.conj(np.transpose(MAdkAdnk, (0, 2, 1)))
        # MBnkBk = M_pi_sub_pairing_BB(k, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell) + np.conj(np.transpose(M_pi_fictitious_Z2_AA(k, 1, A_pi_rs_traced_pp_here, g, unitcell),(0,2,1)))
        # MBdkBdnk = np.conj(np.transpose(MBnkBk,(0,2,1)))

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


def E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
         unitcell=piunitcell, cartesian=False):
    M = M_pi(k, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell, cartesian)
    M = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
    E, V = np.linalg.eigh(M)
    return [E, V]

#endregion

#region find lambda



def I3_integrand(E, lams, Jzz):
    size = int(E.shape[1]/2)
    E = np.sqrt(2*Jzz*(E+np.repeat(lams, size)))
    Ep = Jzz / E
    return np.mean(Ep,axis=1)

def I3_integrand_site(E, V, lams, Jzz, xyz):
    if not xyz:
        E = np.sqrt(2*Jzz*(E+np.repeat(lams, int(E.shape[1]/2))))
    else:
        E = np.sqrt(2*Jzz*(E+np.repeat(np.repeat(lams, int(E.shape[1]/4)),2)))
    Ep = contract('ijk, ijk, ik->ij', V, np.conj(V), Jzz/E)
    return Ep


def rho_true_fast(weights, E, lams, Jzz, xyz):
    return integrate_fixed(I3_integrand, weights, E, lams, Jzz)*np.ones(2)
def rho_true(weights, E, V, lams, Jzz, xyz):
    if not xyz:
        size = int(E.shape[1]/2)
        Ep = I3_integrand_site(E, V, lams, Jzz, xyz)
        lamAl, lamBl = np.mean(Ep[:, 0:size], axis=1), np.mean(Ep[:, size:2 * size], axis=1)
        return np.array([np.real(np.dot(weights, lamAl)), np.real(np.dot(weights, lamBl))])
    else:
        size = int(E.shape[1]/4)
        Ep = I3_integrand_site(E, V, lams, Jzz, xyz)
        lamAl1, lamBl1 = np.mean(Ep[:, 0:size], axis=1), np.mean(Ep[:, size:2 * size], axis=1)
        lamAl2, lamBl2 = np.mean(Ep[:, 2*size:3*size], axis=1), np.mean(Ep[:, 3*size:4 * size], axis=1)
        return np.array([np.real(np.dot(weights, lamAl1)+np.dot(weights, lamAl2)), np.real(np.dot(weights, lamBl1)+np.dot(weights, lamBl2))])


def rho_true_site(weights, E, V, lams, Jzz, xyz):
    return integrate_fixed(I3_integrand_site, weights, E, V, lams, Jzz, xyz)

#endregion

#region gradient find minLam
def Emin(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
         unitcell):
    k = k.reshape((1,3))
    return E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
                unitcell)[0][0,0]


#region find minlam scipy
def findminLam_scipy(M, K, tol, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell, BZres, kappa):
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
                                               A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell),
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
def findlambda_pi(kappa, tol, lamM, Jzz, weights, E, M, xyz=False, inversion=True):
    warnings.filterwarnings("error")
    lamMin = np.copy(lamM)
    lamMax = max(50, 10*lamM[0])*np.ones(2)
    lams = lamMax
    diverge = False
    count = 0
    while True:
        lamlast = np.copy(lams)
        lams = (lamMax+lamMin)/2
        if not inversion:
            if not xyz:
                tempM = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
            else:
                tempM = M + np.diag(np.repeat(np.repeat(lams, int(M.shape[1]/4)),2))
            E, V = np.linalg.eigh(tempM)
        try:
            if inversion:
                rhoguess = rho_true_fast(weights, E, lams, Jzz, xyz)
            else:
                rhoguess = rho_true(weights, E, V, lams, Jzz, xyz)
            error = rhoguess-kappa
            for i in range(2):
                if error[i] > 0:
                    lamMin[i] = lams[i]
                else:
                    lamMax[i] = lams[i]
            if ((np.absolute(rhoguess - kappa) <= tol).all()):
                break
                # print(rhoguess, lams, lamMin, lamMax, abs(lamMin - lamMax), count)
        except:
            lamMin = lams
        count = count + 1
        if (abs(lamMin - lamMax) < 5e-15).all() or count > 1e2:
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

def chiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, chi_field, *args):
    k = contract('ij,jk->ik', pts, BasisBZA)
    A, B = chi_integrand(k, E, V, Jzz, unitcellGraph)
    A = contract('ikjl,i->kjl', A, weights)
    B = contract('ikjl,i->kjl', B, weights)
    M1 = chi_field(n, n1, n2, unitcellCoord, B, A, *args)
    return M1

def xi_integrand(k, E, V, Jzz, unitcellGraph):
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jk->ij', k, NN)
    ffactA = np.exp(1j * ffact)
    size = int(E.shape[1]/4)
    A = contract('ika, ij,jka->ikj', green[:, 0:size, size:2*size], ffactA, unitcellGraph)/size
    return A
def xiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, xi_field, *args):
    k = contract('ij,jk->ik', pts, BasisBZA)
    M = contract('ikj, i->kj', xi_integrand(k,E,V,Jzz,unitcellGraph), weights)
    M1 = xi_field(n, n1, n2, unitcellCoord, M, *args)
    return M1

def calmeanfield(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, xi_field, chi_field, params):
    chi = chiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, chi_field, params)
    # chi = np.zeros((len(unitcellCoord),4,4))
    return chi, xiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, xi_field, params)

def xiCalCondensed(rhos, qmin, n, n1, n2, unitcellCoord, unitcellGraph, xi_field, *args):
    k = contract('ij,jk->ik', qmin, BasisBZA)
    ffact = contract('ik,jk->ij', k, NN)
    ffactA = np.exp(1j * ffact)
    size = int(len(rhos)/4)
    A = contract('k, a, ij,jka->kj', np.conj(rhos[0:size]), rhos[size:2*size], ffactA, unitcellGraph)/size
    # M1 = xi_field(n, n1, n2, unitcellCoord, A, *args)
    return A

def chiCalCondensed(rhos, qmin, n, n1, n2, unitcellCoord, unitcellGraph, chi_field, *args):
    k = contract('ij,jk->ik', qmin, BasisBZA)
    size = int(len(rhos)/4)
    ffact = contract('ik,jlk->ijl', k, NNminus)
    ffactB = np.exp(-1j * ffact)
    B = contract('a, b, ijl,jka, lkb->kjl', rhos[3*size:4*size], rhos[size:2*size], ffactB, unitcellGraph, unitcellGraph)/size
    ffactA = np.exp(1j * ffact)
    A = contract('a, b, ijl,jka, lkb->kjl', rhos[2*size:3*size], rhos[0:size], ffactA, unitcellGraph, unitcellGraph)/size
    # M1 = chi_field(n, n1, n2, unitcellCoord, B, A, *args)
    M1 = np.zeros((2, B.shape[0], B.shape[1], B.shape[2]), dtype=np.complex128)
    M1[0] = A
    M1[1] = B
    return M1

# endregion

#region graphing BZ

def dispersion_pi(lams, k, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, g, unitcell):
    temp = np.sqrt(2 * Jzz * E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                  A_pi_rs_traced_pp_here, g, unitcell, True)[0])
    return temp


def calDispersion(lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
                  axes, g, unitcell=piunitcell):
    dGammaX = dispersion_pi(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                            A_pi_rs_traced_pp_here, g, unitcell)
    dXW = dispersion_pi(lams, XW, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                        A_pi_rs_traced_pp_here, g, unitcell)
    dWK = dispersion_pi(lams, WK, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                        A_pi_rs_traced_pp_here, g, unitcell)
    dKGamma = dispersion_pi(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                            A_pi_rs_traced_pp_here, g, unitcell)
    dGammaL = dispersion_pi(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                            A_pi_rs_traced_pp_here, g, unitcell)
    dLU = dispersion_pi(lams, LU, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                        A_pi_rs_traced_pp_here, g, unitcell)
    dUW1 = dispersion_pi(lams, UW1, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                         A_pi_rs_traced_pp_here, g, unitcell)
    dW1X1 = dispersion_pi(lams, W1X1, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                          A_pi_rs_traced_pp_here, g, unitcell)
    dX1Gamma = dispersion_pi(lams, X1Gamma, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                             A_pi_rs_traced_pp_here, g, unitcell)

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
def minCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
           unitcell):
    temp = np.zeros(len(q))
    mins = dispersion_pi(lams, K, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                         A_pi_rs_traced_pp_here, g, unitcell)[:, 0]
    for i in range(len(q)):
        temp[i] = np.min(
            dispersion_pi(lams, K - q[i], Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                          A_pi_rs_traced_pp_here, g, unitcell)[:, 0]
            + mins)
    return temp


# @nb.njit(parallel=True, cache=True)
def maxCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
           unitcell):
    temp = np.zeros(len(q))
    maxs = dispersion_pi(lams, K, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                         A_pi_rs_traced_pp_here, g, unitcell)[:, -1]
    for i in range(len(q)):
        temp[i] = np.max(
            dispersion_pi(lams, K - q[i], Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                          A_pi_rs_traced_pp_here, g, unitcell)[:, -1]
            + maxs)
    return temp


def minMaxCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
              g, unitcell):
    temp = np.zeros((len(q), 2))
    Ek = dispersion_pi(lams, K, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                       A_pi_rs_traced_pp_here, g, unitcell)
    mins = Ek[:,0]
    maxs = Ek[:,-1]
    for i in range(len(q)):
        tt = dispersion_pi(lams, K - q[i], Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                           A_pi_rs_traced_pp_here, g, unitcell)
        temp[i, 0] = np.min(tt[:, 0] + mins)
        temp[i, 1] = np.max(tt[:, -1] + maxs)
    return temp

def DSSF_E_Low(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
               A_pi_rs_traced_pp_here, g):
    Eq = np.sqrt(2 * Jzz * E_pi(K, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here, g)[0])
    Ek = np.sqrt(2 * Jzz * E_pi(K - q, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here, g)[0])
    return min(Eq[:,0]+Ek[:,0])

def DSSF_E_High(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                A_pi_rs_traced_pp_here, g):
    Eq = np.sqrt(2 * Jzz * E_pi(K, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here, g)[0])
    Ek = np.sqrt(2 * Jzz * E_pi(K - q, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here, g)[0])
    return max(Eq[:,-1]+Ek[:,-1])

def DSSF_E_DOMAIN(lams, qmin, qmax, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, g):
    return DSSF_E_Low(lams, qmin, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                      A_pi_rs_traced_pp_here, g) \
        , DSSF_E_High(lams, qmax, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                      A_pi_rs_traced_pp_here, g)


def loweredge(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
              unitcell, ax, color='w'):
    dGammaX = minCal(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dXW = minCal(lams, XW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dWK = minCal(lams, WK, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dLU = minCal(lams, LU, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dUW1 = minCal(lams, UW1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, g, unitcell)
    dW1X1 = minCal(lams, W1X1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                   A_pi_rs_traced_pp_here, g, unitcell)
    dX1Gamma = minCal(lams, X1Gamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                      A_pi_rs_traced_pp_here, g, unitcell)

    ax.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, color,zorder=8)
    ax.plot(np.linspace(gX, gW, len(dXW)), dXW, color,zorder=8)
    ax.plot(np.linspace(gW, gK, len(dWK)), dWK, color,zorder=8)
    ax.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, color,zorder=8)
    ax.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, color,zorder=8)
    ax.plot(np.linspace(gL, gU, len(dLU)), dLU, color,zorder=8)
    ax.plot(np.linspace(gU, gW1, len(dUW1)), dUW1, color,zorder=8)
    ax.plot(np.linspace(gW1, gX1, len(dW1X1)), dW1X1, color,zorder=8)
    ax.plot(np.linspace(gX1, gGamma3, len(dX1Gamma)), dX1Gamma, color,zorder=8)

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

def upperedge(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
              unitcell, ax, color='w'):
    dGammaX = maxCal(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dXW = maxCal(lams, XW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dWK = maxCal(lams, WK, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dKGamma = maxCal(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dGammaL = maxCal(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dLU = maxCal(lams, LU, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dUW1 = maxCal(lams, UW1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, g, unitcell)
    dW1X1 = maxCal(lams, W1X1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                   A_pi_rs_traced_pp_here, g, unitcell)
    dX1Gamma = maxCal(lams, X1Gamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                      A_pi_rs_traced_pp_here, g, unitcell)


    ax.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, color,zorder=8)
    ax.plot(np.linspace(gX, gW, len(dXW)), dXW, color,zorder=8)
    ax.plot(np.linspace(gW, gK, len(dWK)), dWK, color,zorder=8)
    ax.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, color,zorder=8)
    ax.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, color,zorder=8)
    ax.plot(np.linspace(gL, gU, len(dLU)), dLU, color,zorder=8)
    ax.plot(np.linspace(gU, gW1, len(dUW1)), dUW1, color,zorder=8)
    ax.plot(np.linspace(gW1, gX1, len(dW1X1)), dW1X1, color,zorder=8)
    ax.plot(np.linspace(gX1, gGamma3, len(dX1Gamma)), dX1Gamma, color,zorder=8)

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
def loweredge_data(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                   A_pi_rs_traced_pp_here, g, unitcell):
    dGammaX = minCal(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dXW = minCal(lams, XW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dWK = minCal(lams, WK, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dLU = minCal(lams, LU, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dUW1 = minCal(lams, UW1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, g, unitcell)
    dW1X1 = minCal(lams, W1X1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                   A_pi_rs_traced_pp_here, g, unitcell)
    dX1Gamma = minCal(lams, X1Gamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                      A_pi_rs_traced_pp_here, g, unitcell)

    return np.concatenate((dGammaX, dXW, dWK, dKGamma, dGammaL, dLU, dUW1, dW1X1, dX1Gamma))

def upperedge_data(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                   A_pi_rs_traced_pp_here, g, unitcell):
    dGammaX = maxCal(lams, GammaX, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dXW = maxCal(lams, XW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dWK = maxCal(lams, WK, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dKGamma = maxCal(lams, KGamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dGammaL = maxCal(lams, GammaL, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                     A_pi_rs_traced_pp_here, g, unitcell)
    dLU = maxCal(lams, LU, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, g, unitcell)
    dUW1 = maxCal(lams, UW1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, g, unitcell)
    dW1X1 = maxCal(lams, W1X1, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                   A_pi_rs_traced_pp_here, g, unitcell)
    dX1Gamma = maxCal(lams, X1Gamma, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                      A_pi_rs_traced_pp_here, g, unitcell)

    return np.concatenate((dGammaX, dXW, dWK, dKGamma, dGammaL, dLU, dUW1, dW1X1, dX1Gamma))


#endregion

#region greens function and energetics
def green_pi(E, V, Jzz, omega=0):
    green = contract('ilk, ijk, ik->ijl', V, np.conj(V), Jzz / (omega**2+E))
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
    unitCellgraph = np.array([[[1]],[[1]],[[1]],[[1]]])
    A_pi_here = np.array([[0,0,0,0]])
    unitcellCoord = np.array([[0,0,0]])
    if (flux == np.zeros(4)).all():
        unitCellgraph = np.array([[[1]],[[1]],[[1]],[[1]]])
        A_pi_here = np.array([[0,0,0,0]])
        unitcellCoord = np.array([[0,0,0]])
    elif (flux == np.pi*np.ones(4)).all():
        unitCellgraph = piunitcell
        A_pi_here = A_pi
        unitcellCoord = np.array([[0, 0, 0],[0,1,0],[0,0,1],[0,1,1]])
    elif (flux == pzzp).all():
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
    elif (flux == zppz).all():
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

    return unitCellgraph, A_pi_here, unitcellCoord

def graphing_M_setup_full(flux, n):
    unitCellgraph = piunitcell
    unitcellCoord = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])
    A_pi_here = np.zeros((4,4))
    if (flux == np.zeros(4)).all():
        A_pi_here = np.zeros((4,4))
    elif (flux == np.pi*np.ones(4)).all():
        A_pi_here = A_pi
    elif (flux == pzzp).all():
        A_pi_here = np.array([[0,0,0,0],
                              [0,np.pi,0,0],
                              [0, 0, 0, 0],
                              [0, np.pi, 0, 0]
                              ])
    elif (flux == zppz).all():
        A_pi_here = np.array([[0,0,0,0],
                              [0, 0, 0, 0],
                              [0,np.pi,np.pi,0],
                              [0,np.pi,np.pi,0]])

    return unitCellgraph, A_pi_here, unitcellCoord

def xi_unconstrained(n, n1, n2, unitcellcoord, xi0, args):
    #in the case of 110, three xi mf: xi0, xi1, xi3
    return xi0
def chi_unconstrained(n, n1, n2, unitcellCoord, chi0, chi0A, *args):
    return np.array([chi0, chi0A])
def xi_wo_field(n, n1, n2, unitcellcoord, xi, args):
    #in the case of 110, three xi mf: xi0, xi1, xi3
    xi00 = np.real(np.mean(np.abs(xi)))
    mult = np.zeros((len(unitcellcoord),4),dtype=np.complex128)
    try:
        nS, = args
    except:
        nS = 0
    for i in range(len(unitcellcoord)):
        mult[i] = np.array([xi00, xi00*np.exp(1j*np.pi*(nS+n1*(unitcellcoord[i,1]+unitcellcoord[i,2]))), xi00*np.exp(1j*np.pi*(nS+n1*unitcellcoord[i,2])), xi00*np.exp(1j*np.pi*nS)])
    return mult
def chi_wo_field(n, n1, n2, unitcellCoord, chi, chiA, *args):
    chi0 = chi[0]
    mult = np.zeros((2, len(unitcellCoord),4, 4),dtype=np.complex128)

    chi000 = chi0[0,0]
    chi001 = chi0[0,1]

    chi000 = np.sign(np.real(chi000))*chi000
    chi001 = np.sign(np.real(chi001))*chi001

    for i in range(len(unitcellCoord)):
        r2 = unitcellCoord[i,1]
        r3 = unitcellCoord[i,2]

        chi00 = chi000
        chi01 = chi001*np.exp(1j*np.pi*n1*(r2+r3))
        chi02 = chi001*np.exp(1j*np.pi*(n1*r3))
        chi03 = chi001
        chi12 = chi001*np.exp(1j*np.pi*n1*(r2+1))
        chi13 = chi001*np.exp(1j*np.pi*n1*(r2+r3+1))
        chi23 = chi001*np.exp(1j*np.pi*n1*(r3+1))

        mult[1, i] = np.array([[chi00, chi01, chi02, chi03],
                               [chi01, chi00, chi12, chi13],
                               [chi02, chi12, chi00, chi23],
                               [chi03, chi13, chi23, chi00]])
        mult[0, i] = np.array([[chi00, chi01, chi02, chi03],
                               [chi01, chi00, chi12, chi13],
                               [chi02, chi12, chi00, chi23],
                               [chi03, chi13, chi23, chi00]])
    return mult
def xi_w_field_Octu(n, n1, n2, unitcellcoord, xi, args):
    #in the case of 110, three xi mf: xi0, xi1, xi3
    xi0 = np.real(xi[0])
    mult = np.zeros((len(unitcellcoord),4),dtype=np.complex128)
    for i in range(len(unitcellcoord)):
        if (n==h110).all():
            mult[i] = np.array([xi0[0], xi0[1]*np.exp(1j*(n1*np.pi*unitcellcoord[i,1]+n2*np.pi*unitcellcoord[i,2])), xi0[2]*np.exp(1j*(n2*np.pi*unitcellcoord[i,2])), xi0[0]])
        elif (n==h111).all():
            mult[i] = np.array([xi0[0], xi0[1]*np.exp(1j*(n1*np.pi*unitcellcoord[i,1]+n1*np.pi*unitcellcoord[i,2])), xi0[1]*np.exp(1j*(n1*np.pi*unitcellcoord[i,2])), xi0[1]])
        else:
            xi00 = np.real(np.mean(np.abs(xi)))
            mult[i] = np.array([xi00, xi00*np.exp(1j*(n1*np.pi*unitcellcoord[i,1]+n1*np.pi*unitcellcoord[i,2])), xi00*np.exp(1j*(n1*np.pi*unitcellcoord[i,2])), xi00])
    return mult

def chi_w_field_Octu(n, n1, n2, unitcellCoord, chi, chiA, *args):
    chi0 = chi[0]
    chi0A = chiA[0]
    mult = np.zeros((2, len(unitcellCoord),4, 4),dtype=np.complex128)
    warnings.filterwarnings('error')
    for i in range(len(unitcellCoord)):
        r2 = unitcellCoord[i,1]
        r3 = unitcellCoord[i,2]

        chi00 = chi0[0,0]

        if (n==h110).all():
            try:
                psiI = chi0[0,0]/chi0A[0,0]
                psiIT1 = chi0A[0,1]/chi0[0,1]*psiI
                psiIT2 = chi0A[0,2]/chi0[0,2]*psiI
            except:
                psiIT1, psiIT2, psiI = 1, 1, 1

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
            try:
                psiC6 = chi0[0,0]/chi0A[0,0]
            except:
                psiC6 = 1

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
            try:
                # psiS, = args
                psiS = chi0[2,3]/chi0A[2,3]
            except:
                psiS = 1

            chi01 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi02 = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))/psiS**(3/2)
            chi03 = chi0[0,3]
            chi12 = chi0[0,3]*np.exp(1j*np.pi*(n1*r2))/psiS**(1/2)
            chi13 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))/psiS**(1/2)
            chi23 = chi0[0,1]*np.exp(1j*np.pi*n1*r3)/psiS

            chi00A = chi0[0,0]
            chi01A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi02A = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))/psiS**(3/2)
            chi03A = chi0[0,3]
            chi12A = chi0[0,3]*np.exp(1j*np.pi*(n1*r2))/psiS**(1/2)
            chi13A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))/psiS**(1/2)
            chi23A = chi0[0,1]*np.exp(1j*np.pi*n1*r3)/psiS

        mult[1, i] = np.array([[chi00, chi01, chi02, chi03],
                               [chi01, chi00, chi12, chi13],
                               [chi02, chi12, chi00, chi23],
                               [chi03, chi13, chi23, chi00]])
        mult[0, i] = np.array([[chi00A, chi01A, chi02A, chi03A],
                               [chi01A, chi00A, chi12A, chi13A],
                               [chi02A, chi12A, chi00A, chi23A],
                               [chi03A, chi13A, chi23A, chi00A]])
    warnings.resetwarnings()
    return mult

def xi_w_field_Diu(n, n1, n2, unitcellcoord, xi, args):
    xi0 = np.real(xi[0])
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

def chi_w_field_Diu(n, n1, n2, unitcellCoord, chi, chiA, args):
    chi0 = chi[0]
    chi0A = chiA[0]
    mult = np.zeros((2, len(unitcellCoord),4, 4),dtype=np.complex128)
    warnings.filterwarnings('error')
    for i in range(len(unitcellCoord)):
        r2 = unitcellCoord[i,1]
        r3 = unitcellCoord[i,2]

        chi00 = chi0[0,0]
        if (n==h110).all():
            try:
                psiI = chi0[0,0]/chi0A[0,0]
                nI = (-np.sign(chi0[0,1]/chi0A[0,1]*psiI)+1)/2
                psisigmaT1 = chi0[1,3]/chi0[0,1]
                psisigmaT2 = chi0[2,3]/chi0[1,2]/psisigmaT1
            except:
                psisigmaT1, psisigmaT2, psiI, nI = 1, 1, 1, 0

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
            try:
                # psiC6, = args
                psiC6 = chi0[0,0]/chi0A[0,0]
            except:
                psiC6 = 1

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
            try:
                psiS = chi0[2,3]/chi0A[0,1]
                nI = 0
            except:
                psiS, nI = 1, 0

            chi01 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi02 = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))/psiS**(3/2)
            chi03 = chi0[0,3]
            chi12 = chi0[0,3]*np.exp(1j*np.pi*(n1*r2))/psiS**(1/2)
            chi13 = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))/psiS**(1/2)
            chi23 = chi0[0,1]*np.exp(1j*np.pi*n1*r3)/psiS

            chi00A = chi0[0,0]
            chi01A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))
            chi02A = chi0[0,1]*np.exp(1j*np.pi*(n1*r3))/psiS**(3/2)
            chi03A = chi0[0,3]
            chi12A = chi0[0,3]*np.exp(1j*np.pi*(n1*r2))/psiS**(1/2)
            chi13A = chi0[0,1]*np.exp(1j*np.pi*(n1*r2+n1*r3))/psiS**(1/2)
            chi23A = chi0[0,1]*np.exp(1j*np.pi*n1*r3)/psiS

        mult[1, i] = np.array([[chi00, chi01, chi02, chi03],
                               [chi01, chi00, chi12, chi13],
                               [chi02, chi12, chi00, chi23],
                               [chi03, chi13, chi23, chi00]])
        mult[0, i] = np.array([[chi00A, chi01A, chi02A, chi03A],
                               [chi01A, chi00A, chi12A, chi13A],
                               [chi02A, chi12A, chi00A, chi23A],
                               [chi03A, chi13A, chi23A, chi00A]])
    warnings.resetwarnings()
    return mult

#endregion
class piFluxSolver:
    def __init__(self, Jxx, Jyy, Jzz, *args, theta=0, h=0, n=h110, kappa=2, lam=2, BZres=20, graphres=20, tol=1e-10, flux=np.zeros(4),
                 intmethod=gauss_quadrature_3D_pts, gzz=2.24, Breal=False, unconstrained=False, g=0, simplified=False, FF=False):
        self.intmethod = intmethod
        J = np.array([Jxx, Jyy, Jzz])
        a = np.argmax(J)
        xx = np.mod(a+1,3)
        yy = np.mod(a+2,3)
        self.dominant = a
        self.Jzz = J[a]
        self.Jpm = -(J[xx] + J[yy]) / 4
        self.Jpmpm = (J[xx] - J[yy]) / 4
        self.theta = theta
        self.kappa = kappa
        self.g = g
        self.tol = tol
        self.lams = np.array([lam, lam], dtype=np.double)
        self.PSGparams = args
        if unconstrained:
            self.xi_field = xi_unconstrained
            self.chi_field = chi_unconstrained
        elif h == 0:
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
        self.inversion = True
        # if FF == True:
        #     self.inversion = False
        self.pts, self.weights = self.intmethod(0, 1, 0, 1, 0, 1, BZres, BZres, BZres)

        self.minLams = np.zeros(2, dtype=np.double)
        self.BZres = BZres
        self.graphres = graphres

        self.toignore = np.array([], dtype=int)
        self.q = np.nan
        self.qmin = np.empty((1, 3))
        self.qmin[:] = 0
        self.qminWeight = np.zeros((1,))
        self.qminB = np.copy(self.qmin)
        self.condensed = False
        if not FF:
            self.n = n
            self.flux = flux
            self.A_pi_here, self.n1, self.n2 = determineEquivalence(n, flux)


            # self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.A_pi_rs_rsp_here, self.A_pi_rs_rsp_pp_here = gen_gauge_configurations(self.A_pi_here)
            # self.unitCellgraph = piunitcell
            if simplified:
                self.unitCellgraph, self.A_pi_here, self.unitcellCoord = graphing_M_setup(self.flux, self.n)
            else:
                self.unitCellgraph, self.A_pi_here, self.unitcellCoord = graphing_M_setup_full(self.flux, self.n)
            self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.A_pi_rs_rsp_here, self.A_pi_rs_rsp_pp_here = gen_gauge_configurations(
                self.A_pi_here)
            self.xi = self.xi_field(n, self.n1, self.n2, self.unitcellCoord, np.random.rand(len(self.A_pi_here),4), self.PSGparams)
            self.chi = self.chi_field(n, self.n1, self.n2, self.unitcellCoord, np.random.rand(len(self.A_pi_here),4,4), np.random.rand(len(self.A_pi_here),4,4), self.PSGparams)
            self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                           self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
            self.E, self.V = np.linalg.eigh(self.MF)
            self.delta = np.zeros(self.E.shape[1])
            self.rhos = np.zeros(self.E.shape[1])
        else:
            self.n = h111
            self.flux = flux
            self.unitCellgraph, self.A_pi_here, self.unitcellCoord, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.A_pi_rs_rsp_here, self.A_pi_rs_rsp_pp_here = \
            FFphase_setup(flux[1])
            self.n1, self.n2 = 0, 0
            self.xi = self.xi_field(n, self.n1, self.n2, self.unitcellCoord, 0.0 * np.ones((len(self.A_pi_here), 4)),
                                    self.PSGparams)
            self.chi = self.chi_field(n, self.n1, self.n2, self.unitcellCoord,
                                      0.0 * np.ones((len(self.A_pi_here), 4, 4)),
                                      0.0 * np.ones((len(self.A_pi_here), 4, 4)), self.PSGparams)
            self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                           self.A_pi_here,
                           self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
            self.E, self.V = np.linalg.eigh(self.MF)
            self.delta = np.zeros(self.E.shape[1])
            self.rhos = np.zeros(self.E.shape[1])
        print("Instance Created with parameters " + str(J) + " with flux " + str(flux) + " with external field strength " + str(h) + " n1 = " + str(self.n1) + " n2 = " + str(self.n2))

    def findLambda(self, a=False):
        if a:
            return findlambda_pi(self.kappa, self.tol,self.minLams, self.Jzz, self.weights, self.E, self.MF, (not self.Jpmpm==0), self.inversion)
        else:
            A = -np.min(self.E)*np.ones(2)
            lams, d = findlambda_pi(self.kappa, self.tol, A+1e-16, self.Jzz, self.weights, self.E, self.MF, (not self.Jpmpm==0), self.inversion)
            return lams, d

    def findLambda_unconstrained(self):
        return findlambda_pi(self.kappa,self.tol, np.zeros(2), self.Jzz, self.weights, self.E, self.MF)


    def findminLam(self):
        searchGrid=34
        B = genBZ(searchGrid)
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux, self.n)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(A_pi_here)

        chi = self.chi_field(self.n, self.n1, self.n2, unitcellCoord, self.chi[1], self.chi[0], self.PSGparams)
        xi = self.xi_field(self.n, self.n1, self.n2, unitcellCoord, self.xi, self.PSGparams)

        M = M_pi(B, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi, A_pi_here,
                 A_pi_rs_traced_here, A_pi_rs_traced_pp_here, self.g, unitCellgraph)
        minLams, qmin = findminLam_scipy(M, B, self.tol, self.Jpm, self.Jpmpm, self.h, self.n,
                                        self.theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
                                        self.g, unitCellgraph, searchGrid, self.kappa)
        self.qmin = qmin
        self.qminB = contract('ij,jk->ik', self.qmin, BasisBZA)
        # self.qminWeight = np.ones((len(self.qmin),))/(len(self.pts)+len(self.qmin))
        self.minLams = np.ones(2) * (minLams)
        return minLams

    def rho(self,lam):
        return rho_true(self.weights, self.E, lam,self.Jzz)

    def rho_site(self):
        return rho_true_site(self.weights, self.E, self.V, self.lams,self.Jzz, (not self.Jpmpm==0))

    def calmeanfield(self):
        E, V = self.LV_zero(np.concatenate((self.pts,self.qmin)))
        E = np.sqrt(2*self.Jzz*E)
        chi, xi = calmeanfield(E, V, self.Jzz, self.n, self.n1, self.n2, np.concatenate((self.pts,self.qmin)), np.concatenate((self.weights, self.qminWeight)), self.unitcellCoord, self.unitCellgraph, self.xi_field, self.chi_field, self.PSGparams)
        return chi, xi

    def solvexifield(self):
        E = np.sqrt(2*self.Jzz*(self.E+np.repeat(np.repeat(self.lams,int(self.E.shape[1]/4)),2)))
        xi = xiCal(E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.PSGparams)
        # if self.Jpm > -0.175:
        xiC = xiCalCondensed(self.rhos, self.qmin, self.n, self.n1, self.n2, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.PSGparams)
        return xi + xiC
        # else:
        # return xi
    
    def solvechifield(self):
        E = np.sqrt(2*self.Jzz*(self.E+np.repeat(np.repeat(self.lams,int(self.E.shape[1]/4)),2)))
        chi = chiCal(E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.PSGparams)
        # if self.Jpm > -0.175:
        chiC = chiCalCondensed(self.rhos, self.qmin, self.n, self.n1, self.n2, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.PSGparams)
        return chi + chiC
        # else:
        # return chi
    
    def updateMF(self):
        self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                      self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
        try:
            self.E, self.V = np.linalg.eigh(self.MF)
        except:
            self.lams = (-np.min(self.E)+1e-14)*np.ones(2)
            self.xi = self.solvexifield()
            self.chi = self.solvechifield()
            self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                          self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
            self.E, self.V = np.linalg.eigh(self.MF)
    def xiSubroutine(self, tol, GS, pcon=False):
        if pcon:
            limit = 5
        else:
            limit = 5
        print("Begin Xi Subroutine")
        count = 0
        pb = False
        while True:
            xilast, GSlast = np.copy(self.xi), GS
            # print("Xi Mean Field Compute")
            self.xi = self.solvexifield()
            # print(self.xi)
            self.updateMF()
            # print("Solve mu field")
            GS, diverge = self.solvemufield()
            if np.abs(GS) > 1e1 or diverge:
                pb = True
            count = count + 1
            if ((abs(self.xi - xilast) < tol).all()) or count > limit:
                break
        print("Xi Subrountine ends. Exiting Energy is: "+ str(GS) + " Took " + str(count) + " cycles.")
        return GS, pb
    def chiSubroutine(self, tol, GS, pcon=False):
        if pcon:
            limit = 5
        else:
            limit = 5
        print("Begin Chi Subroutine")
        count = 0
        pb = False
        while True:
            chilast, GSlast = np.copy(self.chi), GS
            # print("Chi Mean Field Compute")
            self.chi = self.solvechifield()
            # print(self.chi[0,0,0,0], self.chi[0,0,0,1])
            self.updateMF()
            # print("Solve mu field")
            GS, diverge = self.solvemufield()
            if np.abs(GS) > 1e1 or diverge:
                pb = True
                # return GSlast, True
            # print(self.chi[0,0,0,0], self.chi[0,0,0,1], GS)
            count = count + 1
            if ((abs(self.chi - chilast) < tol).all()) or count > limit:
                break
        print("Chi Subrountine ends. Exiting Energy is: "+ str(GS) + " Took " + str(count) + " cycles.")
        return GS, pb

    def solvemufield(self, a=True):
        if a:
            self.findminLam()
        self.lams, diverge = self.findLambda(a)
        if a:
            self.set_condensed()
            if self.condensed:
                self.set_delta()
            else:
                self.rhos[:] = 0
        else:
            self.rhos[:] = 0
        return self.MFE(), diverge

    def solvemeanfield(self, all=True, Fast=False, ref_energy=0):
        if Fast:
            self.solvemeanfield_fast(ref_energy)
        else:
            if all:
                return self.solvemeanfield_all()
            else:
                return self.solvemeanfield_seq()
    def solvemeanfield_seq(self, tol=1e-13):
        warnings.filterwarnings('error')
        tstart = time.time()
        if self.Jpmpm == 0 and self.Jpm==0 and self.h==0:
            self.chi = np.zeros((2,len(self.unitcellCoord),4,4))
            self.xi = np.zeros((len(self.unitcellCoord),4))
            self.condensation_check()
            self.condensed = False
        elif self.Jpmpm == 0:
            self.chi = np.zeros((2,len(self.unitcellCoord),4,4))
            self.xi = np.zeros((len(self.unitcellCoord),4))
            self.condensation_check()
        else:
            print("Initialization Routine")
            limit = 10
            GS, d = self.solvemufield()
            print("Initialization Routine Ends. Starting Parameters: GS="+ str(GS) + " xi0= " + str(self.xi[0]) + " chi0= " + str(self.chi[0,0]))
            count = 0
            pcon = False
            while True:
                chilast, xilast, GSlast = np.copy(self.chi), np.copy(self.xi), np.copy(GS)
                GS, pcon = self.xiSubroutine(tol, GS, pcon)
                GS, pcon = self.chiSubroutine(tol, GS, pcon)
                print("Iteration #"+str(count))
                count = count + 1
                if (((abs(self.chi-chilast) < tol).all()) and ((abs(self.xi-xilast) < tol).all())) or count > limit:
                    break
            self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                           self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
            self.E, self.V = np.linalg.eigh(self.MF)
            self.condensation_check()
            print("Finished Solving. Parameters: Jzz=" + str(self. Jzz) + "; Jpm="+str(self.Jpm)+"; Jpmpm="+str(self.Jpmpm)+"; condensed="+str(self.condensed))
        tend = time.time()
        print("This run took "+ str(tend-tstart))
        warnings.resetwarnings()
        return 0
    def solvemeanfield_all(self, tol=1e-8):
        warnings.filterwarnings('error')
        tstart = time.time()
        if self.Jpmpm == 0 and self.Jpm==0 and self.h==0:
            self.chi = np.zeros((2,len(self.unitcellCoord),4,4))
            self.xi = np.zeros((len(self.unitcellCoord),4))
            self.condensation_check()
        elif self.Jpmpm == 0:
            self.chi = np.zeros((2,len(self.unitcellCoord),4,4))
            self.xi = np.zeros((len(self.unitcellCoord),4))
            self.condensation_check()
            print(self.lams)
        else:
            print("Initialization Routine")
            limit = 50
            # hascondensedcount = 8
            GS, d = self.solvemufield()

            print("Initialization Routine Ends. Starting Parameters: GS="+ str(GS) + " xi0= " + str(self.xi[0]) + " chi0= " + str(self.chi[0,0]))
            count = 0
            # pcon = False
            while True:
                chilast, xilast, GSlast = np.copy(self.chi), np.copy(self.xi), np.copy(GS)
                # E = np.sqrt(2 * self.Jzz * (self.E + np.repeat(self.lams, int(self.E.shape[1] / 2))))
                self.xi = self.solvexifield()
                self.chi = self.solvechifield()
                self.updateMF()
                GS, diverge = self.solvemufield()
                # if diverge:
                #     self.inversion = False
                # else:
                #     self.inversion = True
                print("Iteration #"+str(count), GS, self.condensed)
                count = count + 1
                if (((abs(self.chi-chilast) < tol).all()) and ((abs(self.xi-xilast) < tol).all())) or count > limit:
                    break
            self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                           self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
            self.E, self.V = np.linalg.eigh(self.MF)
            self.condensation_check()
            print("Finished Solving. Parameters: Jzz=" + str(self. Jzz) + "; Jpm="+str(self.Jpm)+"; Jpmpm="+str(self.Jpmpm)+"; h="+str(self.h)+"; condensed="+str(self.condensed))
        tend = time.time()
        print("This run took "+ str(tend-tstart))
        warnings.resetwarnings()
        return 0

    def solvemeanfield_fast(self, ref_energy, tol=1e-8):
        warnings.filterwarnings('error')
        tstart = time.time()
        if self.Jpmpm == 0 and self.Jpm == 0 and self.h == 0:
            self.chi = np.zeros((2, len(self.unitcellCoord), 4, 4))
            self.xi = np.zeros((len(self.unitcellCoord), 4))
            self.condensation_check()
        elif self.Jpmpm == 0:
            self.chi = np.zeros((2, len(self.unitcellCoord), 4, 4))
            self.xi = np.zeros((len(self.unitcellCoord), 4))
            self.condensation_check()
        else:
            print("Initialization Routine")
            limit = 100
            GS, d = self.solvemufield(False)
            print("Initialization Routine Ends. Starting Parameters: GS=" + str(GS) + " xi0= " + str(
                self.xi[0]) + " chi0= " + str(self.chi[0, 0]))
            count = 0
            while True:
                chilast, xilast, GSlast = np.copy(self.chi), np.copy(self.xi), np.copy(GS)
                # E = np.sqrt(2 * self.Jzz * (self.E + np.repeat(self.lams, int(self.E.shape[1] / 2))))
                self.xi = self.solvexifield()
                self.chi = self.solvechifield()
                self.updateMF()
                GS, diverge = self.solvemufield(False)
                print("Iteration #" + str(count), GS, self.condensed)
                count = count + 1
                if (((abs(self.chi - chilast) < tol).all()) and ((abs(self.xi - xilast) < tol).all())) or count > limit:
                    break
            self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                           self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                           self.unitCellgraph)
            self.E, self.V = np.linalg.eigh(self.MF)
            self.condensation_check()
            if ref_energy - self.GS() > 5e-15:
                self.condensed = True
            print("Finished Solving. Parameters: Jzz=" + str(self.Jzz) + "; Jpm=" + str(self.Jpm) + "; Jpmpm=" + str(
                self.Jpmpm) + "; h=" + str(self.h) + "; condensed=" + str(self.condensed))
        tend = time.time()
        print("This run took " + str(tend - tstart))
        warnings.resetwarnings()
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
        # print(self.lams[0], self.minLams[0], -np.min(self.E))
        if A < (deltamin /(self.BZres**3)) ** 2:
            self.condensed = True
        else:
            self.condensed = False

    def set_delta(self):
        rho = np.sqrt(self.kappa - self.rho_site())
        # self.delta = np.sqrt(self.Jzz/2)/self.rhos**2
        from scipy.linalg import null_space
        M_kc = M_pi(self.qmin, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                       self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
        M_kc = M_kc + np.diag(np.repeat(self.lams, int(M_kc.shape[1]/2)))
        E, V = np.linalg.eigh(M_kc)
        self.rhos = np.linalg.norm(rho) * V[0,:,0]
        self.rhos = self.rhos
    def condensation_check(self):
        self.findminLam()
        self.lams, d = self.findLambda(True)
        self.set_condensed()
        # self.ifcondense()
        self.set_delta()


    def M_true(self, k):
        return M_pi(k, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                    self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)

    def E_pi_mean(self, k):
        if self.Jpmpm == 0:
            E = np.mean(np.sqrt(2 * self.Jzz *(self.E+np.repeat(self.lams, int(self.E.shape[1]/2)))), axis=1)
        else:
            E = np.mean(np.sqrt(2 * self.Jzz *(self.E+np.repeat(np.repeat(self.lams, int(self.E.shape[1]/4)),2))), axis=1)
        return E

    def E_pi(self, k):
        return np.sqrt(2 * self.Jzz *
                       E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                            self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)[0])
    def E_pi_reduced(self, k):
        # unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux, self.n)
        # A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(
        #     A_pi_here)
        # self.xi = self.xi_field(n, self.n1, self.n2, self.unitcellCoord, 0.05*np.ones(4), self.PSGparams)
        # self.chi = self.chi_field(n, self.n1, self.n2, self.unitcellCoord, 0.02*np.ones((4,4)), 0.05*np.ones((4,4)), self.PSGparams)
        return np.sqrt(2 * self.Jzz *
                       E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                            self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                            self.unitCellgraph)[0])


    def dispersion(self, k):
        return dispersion_pi(self.lams, k, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi,
                             self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                             self.unitCellgraph)

    def LV_zero(self, k):
        # print(self.lams, self.minLams, self.lams-self.minLams, np.abs(np.min(self.E)))
        return E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                    self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)

    def GS(self):
        try:
            E = np.dot(self.E_pi_mean(self.pts), self.weights) - self.kappa*self.lams[0]
        except:
            self.lams = (-np.min(self.E)+1e-16)*np.ones(2)
            E = np.dot(self.E_pi_mean(self.pts), self.weights) - self.kappa*self.lams[0]
        # print(self.lams, self.minLams, self.lams-self.minLams, E)
        return E

    def MFE_condensed(self):
        # M_kc = M_pi(self.qmin, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
        #                self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
        # M_kc = M_kc + np.diag(np.repeat(self.lams, int(M_kc.shape[1]/2)))
        # E, V = np.linalg.eigh(M_kc)
        # return -np.mean(contract('i,ki,i->i',self.rhos, E, self.rhos))/(2*self.Jzz)
        return 0
    def MFE(self):
        Ep = self.GS() + self.MFE_condensed()
        return np.real(Ep)

    def graph(self, axes):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                      self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, axes, self.g,
                      self.unitCellgraph)


    def minCal(self, K):
        E = np.sqrt(2 * self.Jzz * (self.E + np.repeat(self.lams, int(self.E.shape[1] / 2))))
        xi = xiCal(E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.PSGparams)
        chi = chiCal(E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.PSGparams)
        return minCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, chi, xi,
                      self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)

    def maxCal(self, K):
        E = np.sqrt(2 * self.Jzz * (self.E + np.repeat(self.lams, int(self.E.shape[1] / 2))))
        xi = xiCal(E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.PSGparams)
        chi = chiCal(E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.PSGparams)
        return maxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, chi, xi,
                      self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)

    def minMaxCal(self, K):
        return minMaxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, self.chi,
                         self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                         self.unitCellgraph)

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
        E = np.sqrt(2 * self.Jzz * (self.E + np.repeat(self.lams, int(self.E.shape[1] / 2))))
        xi = xiCal(E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.PSGparams)
        chi = chiCal(E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.PSGparams)
        q = np.sqrt(2 * self.Jzz *
                    E_pi(B, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi, A_pi_here,
                         A_pi_rs_traced_here, A_pi_rs_traced_pp_here, self.g, unitCellgraph)[0])
        mins = np.min(q[:,0])
        maxs = np.max(q[:,-1])
        return 2*mins, 2*maxs


    def graph_loweredge(self, show, ax=plt, color='w'):
        if show:
            min = loweredge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, contract('ij,jk->ik',self.pts, BasisBZA), self.theta, self.chi, self.xi,
                        self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph,
                        ax, color)
            plt.show()
        else:
            min = loweredge_data(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, contract('ij,jk->ik',self.pts, BasisBZA), self.theta, self.chi, self.xi,
                        self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
        return min

    def graph_upperedge(self, show, ax=plt, color='w'):
        if show:
            max = upperedge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, contract('ij,jk->ik',self.pts, BasisBZA), self.theta, self.chi, self.xi,
                        self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph,
                        ax, color)
            plt.show()
        else:
            max = upperedge_data(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, contract('ij,jk->ik',self.pts, BasisBZA), self.theta, self.chi, self.xi,
                        self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
        return max


    def loweredge(self):
        min = loweredge_data(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, contract('ij,jk->ik',self.pts, BasisBZA), self.theta, self.chi, self.xi,
                             self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                             self.unitCellgraph)
        return min

    def upperedge(self):
        max = upperedge_data(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, contract('ij,jk->ik',self.pts, BasisBZA), self.theta, self.chi, self.xi,
                             self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                             self.unitCellgraph)
        return max

    def green_pi(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi(E, V, self.Jzz)

    def green_pi_branch(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi_branch(E, V, self.Jzz), E

    def green_pi_reduced(self, k, cartesian=False):
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux, self.n)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(
            A_pi_here)
        xi = self.xi_field(self.n, self.n1, self.n2, unitcellCoord, self.xi, self.PSGparams)
        chi = self.chi_field(self.n, self.n1, self.n2, unitcellCoord, self.chi[1], self.chi[0], self.PSGparams)
        E, V = E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi, A_pi_here,
                    A_pi_rs_traced_here, A_pi_rs_traced_pp_here, self.g, unitCellgraph, cartesian)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi(E, V, self.Jzz)

    def green_pi_branch_reduced(self, k, cartesian=False):
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux, self.n)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(
            A_pi_here)
        xi = self.xi_field(self.n, self.n1, self.n2, unitcellCoord, self.xi, self.PSGparams)
        chi = self.chi_field(self.n, self.n1, self.n2, unitcellCoord, self.chi[1], self.chi[0], self.PSGparams)
        E, V = E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi, A_pi_here,
                    A_pi_rs_traced_here, A_pi_rs_traced_pp_here, self.g, unitCellgraph, cartesian)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi_branch(E, V, self.Jzz), E, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here, unitCellgraph


    def order(self):
        if self.condensed and not self.Jpmpm == 0:
            size = int(len(self.rhos)/4)
            return np.mean(contract('i,i->i',np.conj(self.rhos[0:size]),self.rhos[size:2*size]))
        else:
            return 0
    def gap(self):
        return np.sqrt(2*self.Jzz*(np.min(self.E)+self.lams[0]))

    def mag_integrand(self, k):
        # E, V = E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
        #      self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitCellgraph)
        E = np.sqrt(2 * self.Jzz * (self.E+self.lams[0]))
        green = green_pi(E, self.V, self.Jzz)

        ffact = contract('ik, jk->ij', k, NN)
        ffact = np.exp(1j * ffact)
        l = len(self.A_pi_here)
        # zmag = contract('k,ik->i', self.n, z)
        zmag = np.ones(4)
        # magp = contract('ku, u, ru, urx,krx -> kru', ffact, zmag, np.exp(1j * self.A_pi_here), self.unitCellgraph, green[:,0:l,l:2*l])/(l)
        magp = contract('ika, ij,jka, kj->ika', green[:, 0:l, l:2*l], ffact, self.unitCellgraph, np.exp(1j * self.A_pi_here))/l
        # print(np.mean(contract('ika, ij,jka, kj->ika', green[:, 0:l, l:2*l], ffact, self.unitCellgraph, np.exp(1j * self.A_pi_here))/l, axis=0))
        mag = np.real(magp)
        # mag = np.real(contract('ku, u,k->k', ffact * (np.cos(self.theta) - 1j * np.sin(self.theta)), zmag, np.sum(green[:,0:l,l:2*l],axis=(1,2)))/(l))

        # magp = np.real(contract('ku, ru, krx, urx->rku', ffact, np.exp(1j*self.A_pi_here), green[:, 0:l, l:2*l], self.unitCellgraph))
        return mag
    def magnetization(self):
        sz = np.einsum('kru,k->ru',self.mag_integrand(self.pts), self.weights)
        print(sz)
        mag = np.mean(sz)
        if self.condensed:
            mag = np.nan
        return np.real(mag)