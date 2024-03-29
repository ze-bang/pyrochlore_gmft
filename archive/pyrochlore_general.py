import matplotlib.pyplot as plt
import warnings

import numpy as np

from misc_helper import *
from flux_stuff import *
from numpy.testing import assert_almost_equal, assert_allclose

# Here, magnetic field is applied at the 110 direction. In which case,


def M_pi_AA_gen(k, Jpm):
    M = np.zeros((len(k),4,4),dtype=np.complex128)
    for i in range(len(k)):
        kx, ky, kz = k[i]
        M[i] = np.array([[-1/2*(Jpm*np.cos((ky + kz)/2)), -1/2*(Jpm*(np.cos((kx + kz)/2) + \
1.0j*np.sin((kx - ky)/2))), ((1.0j + np.sqrt(3))*Jpm*(np.cos((kx + \
ky)/2) + 1.0j*np.sin((kx - kz)/2)))/4, ((1 - \
1.0j*np.sqrt(3))*Jpm*np.sin((ky - kz)/2))/4], [-1/2*(Jpm*(np.cos((kx \
+ kz)/2) - 1.0j*np.sin((kx - ky)/2))), (Jpm*np.cos((ky + kz)/2))/2, \
(1.0j/4)*(1.0j + np.sqrt(3))*Jpm*np.sin((ky - kz)/2), -1/4*((1.0j + \
np.sqrt(3))*Jpm*(np.cos((kx + ky)/2) - 1.0j*np.sin((kx - kz)/2)))], \
[((-1.0j + np.sqrt(3))*Jpm*(np.cos((kx + ky)/2) - 1.0j*np.sin((kx - \
kz)/2)))/4, -1/4*((1 + 1.0j*np.sqrt(3))*Jpm*np.sin((ky - kz)/2)), \
(Jpm*np.cos((ky + kz)/2))/2, -1/2*(Jpm*(np.cos((kx + kz)/2) - \
1.0j*np.sin((kx - ky)/2)))], [((1 + 1.0j*np.sqrt(3))*Jpm*np.sin((ky - \
kz)/2))/4, -1/4*((-1.0j + np.sqrt(3))*Jpm*(np.cos((kx + ky)/2) + \
1.0j*np.sin((kx - kz)/2))), -1/2*(Jpm*(np.cos((kx + kz)/2) + \
1.0j*np.sin((kx - ky)/2))), -1/2*(Jpm*np.cos((ky + kz)/2))]])
    return M

def M_pi_mag_alg(k, h, n):
    hx, hy, hz = n
    M = np.zeros((len(k),4,4),dtype=np.complex128)
    for i in range(len(k)):
        kx, ky, kz = k[i]
        M[i] = np.array([[((-1)**(7/8)*h*(hx + hy + np.exp((1j/2)*(ky + kz))*(hx - hy - \
hz) + hz))/(4*np.sqrt(3)*np.exp((1j/4)*(kx + ky + kz))), \
-1/4*((-1)**(7/8)*np.exp((1j/4)*(kx - ky + kz))*h*(hx - hy + \
hz))/np.sqrt(3), -1/4*((-1)**(1/24)*np.exp((1j/4)*(kx + ky - \
kz))*h*(hx + hy - hz))/np.sqrt(3), 0], \
[-1/4*((-1)**(7/8)*np.exp((1j/4)*(kx - ky + kz))*h*(hx - hy + \
hz))/np.sqrt(3), ((-1)**(7/8)*h*(hx + hy + hz + np.exp((1j/2)*(ky + \
kz))*(-hx + hy + hz)))/(4*np.sqrt(3)*np.exp((1j/4)*(kx + ky + \
kz))), 0, ((-1)**(1/24)*np.exp((1j/4)*(kx + ky - kz))*h*(hx + hy - \
hz))/(4*np.sqrt(3))], [-1/4*((-1)**(1/24)*np.exp((1j/4)*(kx + ky - \
kz))*h*(hx + hy - hz))/np.sqrt(3), 0, ((-1)**(5/24)*h*(-hx - hy + \
np.exp((1j/2)*(ky + kz))*(hx - hy - hz) - \
hz))/(4*np.sqrt(3)*np.exp((1j/4)*(kx + ky + kz))), \
((-1)**(5/24)*np.exp((1j/4)*(kx - ky + kz))*h*(hx - hy + \
hz))/(4*np.sqrt(3))], [0, ((-1)**(1/24)*np.exp((1j/4)*(kx + ky - \
kz))*h*(hx + hy - hz))/(4*np.sqrt(3)), \
((-1)**(5/24)*np.exp((1j/4)*(kx - ky + kz))*h*(hx - hy + \
hz))/(4*np.sqrt(3)), ((-1)**(5/24)*h*(-hx - hy - hz + \
np.exp((1j/2)*(ky + kz))*(-hx + hy + \
hz)))/(4*np.sqrt(3)*np.exp((1j/4)*(kx + ky + kz)))]])
    return M

#region Hamiltonian Construction
def M_pi_mag_sub_AB(k, h, n, theta, A_pi_here):
    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(-1j * ffact)
    M = contract('ku, u, ru, urx->krx', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                 np.exp(1j*A_pi_here), piunitcell)
    return M


def M_pi_sub_intrahopping_AA(k, alpha, eta, Jpm, A_pi_rs_traced_here):
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

def M_pi_alg_0_flux(k,Jpm):

    M = np.zeros((len(k),4,4), dtype=np.complex128)

    for i in range(len(k)):
        kx = k[i, 0]
        ky = k[i, 1]
        kz = k[i, 2]
        M[i] = np.array([[-1/2*(Jpm*np.cos((ky + kz)/2)), -1/2*(Jpm*(np.cos((kx - ky)/2) + \
        np.cos((kx + kz)/2))), -1/2*(Jpm*(np.cos((kx + ky)/2) + np.cos((kx - \
        kz)/2))), -1/2*(Jpm*np.cos((ky - kz)/2))], [-1/2*(Jpm*(np.cos((kx - \
        ky)/2) + np.cos((kx + kz)/2))), -1/2*(Jpm*np.cos((ky + kz)/2)), \
        -1/2*(Jpm*np.cos((ky - kz)/2)), -1/2*(Jpm*(np.cos((kx + ky)/2) + \
        np.cos((kx - kz)/2)))], [-1/2*(Jpm*(np.cos((kx + ky)/2) + np.cos((kx \
        - kz)/2))), -1/2*(Jpm*np.cos((ky - kz)/2)), -1/2*(Jpm*np.cos((ky + \
        kz)/2)), -1/2*(Jpm*(np.cos((kx - ky)/2) + np.cos((kx + kz)/2)))], \
        [-1/2*(Jpm*np.cos((ky - kz)/2)), -1/2*(Jpm*(np.cos((kx + ky)/2) + \
        np.cos((kx - kz)/2))), -1/2*(Jpm*(np.cos((kx - ky)/2) + np.cos((kx + \
        kz)/2))), -1/2*(Jpm*np.cos((ky + kz)/2))]])

    return M
def M_pi(k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])

    dummy = np.zeros((len(k), 4, 4), dtype=np.complex128)

    MAk = M_pi_sub_intrahopping_AA(k, 0, eta, Jpm, A_pi_rs_traced_here)
    MBk = M_pi_sub_intrahopping_AA(k, 1, eta, Jpm, A_pi_rs_traced_here)
    MAnk = M_pi_sub_intrahopping_AA(-k, 0, eta, Jpm, A_pi_rs_traced_here)
    MBnk = M_pi_sub_intrahopping_AA(-k, 1, eta, Jpm, A_pi_rs_traced_here)


    # MAk = M_pi_AA_gen(k, Jpm)
    # MBk = np.conj(M_pi_AA_gen(k, Jpm))
    # MAnk = M_pi_AA_gen(-k, Jpm)
    # MBnk = np.conj(M_pi_AA_gen(-k, Jpm))
    #
    # MagAkBk = M_pi_mag_alg(k, h, n) + M_pi_sub_interhopping_AB(k, 0, Jpmpm, xi, A_pi_rs_traced_pp_here)
    # MagBkAk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))
    # MagAnkBnk = M_pi_mag_alg(-k, h, n) + M_pi_sub_interhopping_AB(-k, 0, Jpmpm, xi, A_pi_rs_traced_pp_here)
    # MagBnkAnk = np.conj(np.transpose(MagAnkBnk, (0, 2, 1)))

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

def E_pi_fixed(lams, M):
    M = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(M)
    return [E, V]


def E_pi(lams, k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    M = M_pi(k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    M = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(M)

    return [E, V]


def rho_true(Jzz, M, index, mask, dV, lams):
    Mtemp = np.delete(M, index, axis=0)
    masktemp = np.delete(mask, index, axis=0)
    temp = Mtemp + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(temp)
    E = np.sqrt(2*Jzz*E)
    Vt = np.real(contract('ijk,ijk->ijk', V, np.conj(V)))
    Ep = contract('ijk, ik, i->j', Vt, Jzz / E, masktemp)*dV
    return np.mean(Ep) * np.ones(2)

def rho_discrete(Jzz, M, mask, dV, lams):
    temp = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(temp)
    E = np.sqrt(2*Jzz*E)
    Vt = np.real(contract('ijk,ijk->ijk', V, np.conj(V)))
    Ep = contract('ijk, ik->ij', Vt, Jzz / E)
    return np.mean(Ep) * np.ones(2)


def rho_true_site(Jzz, M, index, mask, dV, lams):
    Mtemp = np.delete(M, index, axis=0)
    masktemp = np.delete(mask, index, axis=0)
    temp = Mtemp + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(temp)
    Vt = np.real(contract('ijk,ijk->ijk', V, np.conj(V)))
    Ep = contract('ijk, ik, i->j', Vt, Jzz / np.sqrt(2 * Jzz * E), masktemp)*dV
    return Ep[0:8]

def rho_discrete_site(Jzz, M, mask, dV, lams):
    temp = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(temp)
    E = np.sqrt(2*Jzz*E)
    Vt = np.real(contract('ijk,ijk->ijk', V, np.conj(V)))
    Ep = contract('ijk, ik->ij', Vt, Jzz / E)
    return np.mean(Ep, axis=0)[0:8]




def Emin(q, lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    k = contract('i, ik->k', q, BasisBZA).reshape((1,3))
    return E_pi(lams, k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0][0,0]


def gradient(k, lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    kx, ky, kz = k
    step = 1e-8
    fx = (Emin(np.array([kx + step, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step
    fy = (Emin(np.array([kx, ky + step, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step
    fz = (Emin(np.array([kx, ky, kz + step]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step
    return np.array([fx, fy, fz])

def hessian(k, lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    kx, ky, kz = k
    step = 1e-8

    fxx = (Emin(np.array([kx + step, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx - step, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    fxy = (Emin(np.array([kx, ky + step, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx - step, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    fxz = (Emin(np.array([kx, ky, kz + step]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx - step, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    fyy = (Emin(np.array([kx, ky+ step, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx, ky - step, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    fyz = (Emin(np.array([kx, ky, kz+ step]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx, ky - step, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    fzz = (Emin(np.array([kx, ky, kz + step]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - 2*Emin(
        np.array([kx, ky, kz]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
           + Emin(np.array([kx, ky, kz - step]), lams, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)) / step**2
    return np.array([[fxx, fxy, fxz],[fxy, fyy, fyz],[fxz, fyz, fzz]])

def findminLam(M, K, tol, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, BZres):
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
                gradlen = gradient(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here) - gradient(Klast,
                        np.zeros(2),eta,Jpm,Jpmpm, h, n,theta,chi,chi0,xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
                try:
                    step = abs(np.dot(Know[i] - Klast, gradlen)) / np.linalg.norm(gradlen) ** 2
                except:
                    step = 0

            Klast = np.copy(Know[i])
            Know[i] = Know[i] - step * gradient(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
            Elast = np.copy(Enow[i])
            Enow[i] = Emin(Know[i], np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
            init = False
            if abs(Elast-Enow[i])<1e-12:
                stuff = False
    warnings.resetwarnings()

    a = np.argmin(Enow)
    Know = np.mod(Know[a], 2*np.pi).reshape((1,3))
    # Know = np.where(Know>np.pi, Know-2*np.pi, Know)
    return -Enow[a], Know

def findminLam_scipy(M, K, tol, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, BZres):
    if Jpm==0 and Jpmpm == 0 and h == 0:
        return -1/2, np.array([0,0,0]).reshape((1,3))

    E, V = np.linalg.eigh(M)
    E = np.around(E[:,0], decimals=16)
    Em = E.min()
    dex = np.where(E==Em)
    Know = np.unique(np.around(K[dex], decimals=15), axis=0)
    # print(Know, E[dex])
    # print("--------------------")
    if Know.shape == (3,):
        Know = Know.reshape(1,3)


    Enow = np.zeros(len(Know))

    for i in range(len(Know)):
        res = minimize(Emin, x0=Know[i], args=(np.zeros(2), eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here,
                                               A_pi_rs_traced_here, A_pi_rs_traced_pp_here),
                       method='Nelder-Mead', bounds=((Know[i,0]-1/BZres, Know[i,0]+1/BZres), (Know[i,1]-1/BZres,Know[i,1]+1/BZres), (Know[i,2]-1/BZres,Know[i,2]+1/BZres)))
        Know[i] = np.array(res.x)
        Enow[i] = res.fun
    Enowm = Enow.min()
    dex = np.where(Enow==Enowm)
    # print(Know, Enow)
    Know = np.unique(np.mod(Know[dex], 1),axis=0)
    if Know.shape == (3,):
        Know = Know.reshape(1,3)
    Know = np.array(contract('ij,jk->ik', Know, BasisBZA), order='C')

    return -Enowm, Know

def findlambda_pi(M, index, mask, dV, Jzz, kappa, tol, lamM=np.zeros(2)):
    warnings.filterwarnings("error")
    if lamM[0] == 0:
        lamMin = np.zeros(2)
        lamMax = np.ones(2)
    else:
        lamMin = np.copy(lamM)
        lamMax = np.ones(2)*10
    lams = lamMax
    while True:
        lamlast = np.copy(lams)
        lams = (lamMax+lamMin)/2
        try:
            rhoguess = rho_true(Jzz, M, index, mask, dV, lams)
            for i in range(2):
                if rhoguess[i] - kappa > 0:
                    lamMin[i] = lams[i]
                else:
                    lamMax[i] = lams[i]
            if (abs(lamlast - lams) < 1e-15).all() or ((np.absolute(rhoguess - kappa) <= tol).all()):
                break
        except:
            lamMin = lams
        # print(lams, rhoguess)
    warnings.resetwarnings()
    return lams

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
    chi = chi * np.sign(chi)
    chi0 = chi0 * np.sign(chi0)
    return chi, chi0

def xiCal(lams, M, K, Jzz, ns):
    E, V = E_pi_fixed(lams, M)
    E = np.sqrt(2 * Jzz * E)
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jk->ij', K, NN)
    ffactA = np.exp(1j * ffact)

    M1 = np.mean(contract('ika, ij,jka->ikj', green[:, 0:4, 4:8], ffactA, piunitcell), axis=0)

    M1 = M1[0, 0]
    return np.real(np.abs(M1))

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
    return np.real(np.abs(M1))

def calmeanfieldC(rhos, K):
    chi, chi0 = chiCalC(rhos, K)
    return chi, chi0, xiCalC(rhos, K)

#endregion


# graphing BZ

def dispersion_pi(lams, k, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    temp = np.sqrt(2 * Jzz * E_pi(lams, k, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])
    return temp


def calDispersion(lams, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    dGammaX = dispersion_pi(lams, GammaX, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dXW = dispersion_pi(lams, XW, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dWK = dispersion_pi(lams, WK, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dKGamma = dispersion_pi(lams, KGamma, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dGammaL = dispersion_pi(lams, GammaL, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dLU = dispersion_pi(lams, LU, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dUW = dispersion_pi(lams, UW, Jzz, Jpm, Jpmpm, eta, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)

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
def minCal(lams, q, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    temp = np.zeros(len(q))
    mins = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])[:, 0]
    for i in range(len(q)):
        temp[i] = np.min(
            np.sqrt(2 * Jzz * E_pi(lams, K - q[i], eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])[:, 0] + mins)
    return temp


# @nb.njit(parallel=True, cache=True)
def maxCal(lams, q, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    temp = np.zeros(len(q))
    maxs = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])[:, -1]
    for i in range(len(q)):
        temp[i] = np.max(
            np.sqrt(2 * Jzz * E_pi(lams, K - q[i], eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])[:, -1] + maxs)
    return temp


def minMaxCal(lams, q, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    temp = np.zeros((len(q), 2))
    Ek = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])
    mins = Ek[:,0]
    maxs = Ek[:,-1]
    for i in range(len(q)):
        tt = np.sqrt(2 * Jzz * E_pi(lams, K - q[i], eta, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0])
        temp[i, 0] = np.min(tt[:, 0] + mins)
        temp[i, 1] = np.max(tt[:, -1] + maxs)
    return temp


def loweredge(lams, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    dGammaX = minCal(lams, GammaX, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dXW = minCal(lams, XW, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dWK = minCal(lams, WK, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dLU = minCal(lams, LU, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dUW = minCal(lams, UW, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)

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


def upperedge(lams, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    dGammaX = maxCal(lams, GammaX, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dXW = maxCal(lams, XW, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dWK = maxCal(lams, WK, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dKGamma = maxCal(lams, KGamma, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dGammaL = maxCal(lams, GammaL, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dLU = maxCal(lams, LU, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    dUW = maxCal(lams, UW, Jzz, Jpm, Jpmpm, eta, h, n, K, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)

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


def MFE(Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, M, lams, k, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, mask):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])

    M = M + np.diag(np.repeat(np.repeat(lams, 4), 2))
    E, V = np.linalg.eigh(M)
    E = np.sqrt(2 * Jzz * E)
    green = green_pi(E, V, Jzz)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffactA = np.exp(-1j * ffact)
    ffactB = np.exp(1j * ffact)

    EQ = np.real(np.trace(np.mean(contract('ilk, ijk,ik, i->ijl',  V, np.conj(V) , E / 2, mask),axis=0)) / 2)

    E1A = np.mean(contract('jl,klj, iab, ijl, i, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced_here / 4, green[:, 0:4, 0:4], ffactA, mask,
                 piunitcell, piunitcell))
    E1B = np.mean(contract('jl,klj, iab, ijl, i, jka, lkb->i', notrace, -Jpm * A_pi_rs_traced_here / 4, green[:, 4:8, 4:8], ffactB, mask,
                 piunitcell, piunitcell))

    E1 = np.real(E1A + E1B)

    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(-1j * ffact)
    Emag = np.mean(contract('ku, u, ru, krx, k, urx->k', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                            np.exp(1j*A_pi_here), green[:, 0:4, 4:8], mask, piunitcell), axis=0)

    Emag = 2 * np.real(Emag)

    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    tempxb = xi[1]
    tempxa = xi[0]
    M1a = np.mean(contract('jl, kjl, ij, kl, ikx, i, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, tempxa,
                           green[:, 0:4, 4:8], mask, piunitcell), axis=0)
    M1b = np.mean(contract('jl, kjl, il, kj, ikx, i, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, tempxa,
                           green[:, 0:4, 4:8], mask, piunitcell), axis=0)
    M2a = np.mean(
        contract('jl, kjl, ij, kl, ixk, i, jkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(tempxb),
                 green[:, 0:4, 4:8], mask, piunitcell), axis=0)
    M2b = np.mean(
        contract('jl, kjl, il, kj, ixk, i, lkx->i', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(tempxb),
                 green[:, 0:4, 4:8], mask, piunitcell), axis=0)
    EAB = 2 * np.real(M1a + M1b + M2a + M2b)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    beta = 1
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = np.mean(contract('jl, kjl, kjl, i, ikk->i', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, tempchi, mask, green[:, 0:4, 8:12]),
                 axis=0)

    M2 = np.mean(contract('jl, kjl, ijl, k, iba, i, jka, lkb->i', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0,
                          green[:, 0:4, 8:12], mask, piunitcell, piunitcell), axis=0)

    EAA = 2 * np.real(M1 + M2)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * ffact)
    beta = 0
    tempchi = chi[beta]
    tempchi0 = chi0[beta]

    M1 = np.mean(
        contract('jl, kjl, kjl, i, ikk->i', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, tempchi, mask, green[:, 4:8, 12:16]), axis=0)

    M2 = np.mean(contract('jl, kjl, ijl, k, iba, i, jka, lkb->i', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0,
                          green[:, 4:8, 12:16], mask, piunitcell, piunitcell), axis=0)

    EBB = 2 * np.real(M1 + M2)

    E = EQ + Emag + E1 + EAB + EAA + EBB
    # print(EQ/4, E1/4, Emag/4, EAB/4, EAA/4, EBB/4, E/4)
    return E / 4

def MFE_condensed(Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, M, lams, k, rhos, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
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
    def __init__(self, Jxx, Jyy, Jzz, theta=0, h=0, n=np.array([0, 0, 0]), eta=1, kappa=2, lam=2, BZres=20, graphres=20,
                 ns=1, flux=np.zeros(4)):
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
        self.validgauge = True

        if (n == h110).all():
            self.A_pi_here = constructA_pi_110(flux)
        elif (n == h111).all():
            self.A_pi_here = constructA_pi_111(flux)
        elif (n == h001).all():
            self.A_pi_here = constructA_pi_001(flux)


        self.minLams = np.zeros(2, dtype=np.double)
        self.Mcondensed = np.zeros((16,16))
        self.toignore = np.array([],dtype=int)

        self.BZres = BZres
        self.graphres = graphres
        self.bigB, self.bareB, self.mask, self.dV = genBZ(BZres)

        # self.bigTemp = np.copy(self.bigB)
        self.q = np.nan
        self.qmin = np.empty(3)
        self.qmin[:] = np.nan
        self.condensed = False
        self.delta = np.zeros(2)
        self.rhos = np.zeros(8)

        try:
            self.A_pi_here.shape
        except:
            self.validgauge=False
            return None

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



        self.MF = M_pi(self.bigB, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
                       self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        self.MForig = np.copy(self.MF)


        #goal is to also iterate through the different fluxes. One way to expediate this process is by looking at how
        #magnetic field will affect each hex ring.
        #Ring 012, 123, 230, 301 are perpendicular to
        #[1, 1, -1], [-1,-1,-1], [-1, 1, 1], [1,-1, 1]

    def findLambda(self):
        if not self.validgauge:
            return -1
        return findlambda_pi(self.MF, self.toignore, self.mask, self.dV, self.Jzz, self.kappa, self.tol, self.minLams)

    def fastconverge(self):
        self.lams = findlambda_pi(self.MF, self.toignore, self.mask, self.dV, self.Jzz, self.kappa, self.tol, self.minLams)

    def findminLam(self, chi, chi0, xi):
        if not self.validgauge:
            return -1
        minLams, self.qmin = findminLam_scipy(self.MF, self.bareB, self.tol, self.eta, self.Jpm, self.Jpmpm, self.h, self.n,
                                        self.theta, chi, chi0, xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.BZres)
        self.minLams = np.ones(2) * minLams
        # self.Mcondensed = M_pi(self.qmin, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, chi0, xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        return minLams
    def calmeanfield(self, lams, MF, K):
        if not self.validgauge:
            return -1
        if self.condensed:
            chic, chi0c, xic = calmeanfieldC(self.rhos, self.qmin)
            chi, chi0, xi = calmeanfield(lams, MF, K, self.Jzz, self.ns)
            return chi + chic, chi0 + chi0c, xi + xic
        else:
            chi, chi0, xi = calmeanfield(lams, MF, K, self.Jzz, self.ns)
        return np.array([chi, chi0, xi])

    def solvemeanfield(self, tol=1e-15):
        if not self.validgauge:
            return -1
        mfs = np.array([self.chi, self.chi0, self.xi])
        self.condensation_check(mfs)
        mfs = self.calmeanfield(self.lams, self.MF, self.bigB)
        do = not (self.Jpmpm == 0)
        counter = 0
        while do:
            mfslast = np.copy(mfs)
            self.condensation_check(mfs)
            mfs = self.calmeanfield(self.lams, self.MF, self.bigB)
            print(mfs, self.lams, self.minLams)
            if (abs(mfs-mfslast) < tol).all() or counter >= 30:
                break
            counter = counter + 1
        if do:
            lam, K = self.condensation_check(mfs)
        self.chi, self.chi0, self.xi = mfs
        return 0

    def qvec(self):
        E = \
        E_pi(np.zeros(2), self.bigB, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
             self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)[0][:, 0]
        A = np.where(E == E.min())
        self.q = self.bigB[A]

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

    def set_condensed(self, l):
        A = -self.minLams[0] + self.lams[0]
        # B = (2e2 / len(self.bigB)) ** 2
        self.condensed = A < (deltamin/ l) ** 2
        if self.condensed==True:
            self.toignore = indextoignore(self.bigB, self.qmin)

    def set_delta(self, MF, minLams, lams):
        if self.condensed:
            warnings.filterwarnings('error')
            try:
                self.delta = np.sqrt(lams - minLams) * len(self.bigB)
            except:
                self.delta = np.zeros(2)
            try:
                self.rhos = np.sqrt(self.kappa - rho_true_site(self.Jzz, MF, self.toignore, self.mask, self.dV, lams))
            except:
                self.rhos = np.zeros(8)
            warnings.resetwarnings()
    def condensation_check(self, mfs):
        chi, chi0, xi = mfs
        self.findminLam(chi, chi0, xi)
        self.lams = self.findLambda()
        l = len(self.bigB)
        self.set_condensed(l)
        self.set_delta(self.MF, self.minLams, self.lams)

    def M_true(self, k):
        if not self.validgauge:
            return -1
        return M_pi(k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def E_pi(self, k):
        if not self.validgauge:
            return -1
        return np.sqrt(2 * self.Jzz *
                       E_pi(self.lams, k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi,
                            self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)[0])

    def E_pi_fixed(self):
        if not self.validgauge:
            return -1
        return np.sqrt(2 * self.Jzz * E_pi_fixed(self.lams, self.MF)[0])

    def dispersion(self, k):
        if not self.validgauge:
            return -1
        return dispersion_pi(self.lams, k, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta,
                             self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def LV_zero(self, k, lam=np.zeros(2)):
        if not self.validgauge:
            return -1
        if np.any(lam == 0):
            lam = self.lams
        return E_pi(lam, k, self.eta, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def gap(self):
        if not self.validgauge:
            return -1
        return np.sqrt(2 * self.Jzz * gap(self.MF, self.lams))

    def gapwhere(self):
        if not self.validgauge:
            return -1
        temp = self.MF + np.diag(np.repeat(self.lams, 4))
        E, V = np.linalg.eigh(temp)
        # E = np.sqrt(2*self.Jzz*E)
        dex = np.argmin(E[:, 0])
        return np.mod(self.bigB[dex], 2 * np.pi)


    def mag_con(self):
        if not self.validgauge:
            return -1
        return np.mean(E_pi_fixed(np.zeros(2), self.MF)[0])

    def GS(self):
        if not self.validgauge:
            return -1
        return np.mean(self.E_pi(self.bigB)) - np.mean(self.lams)


    def MFE(self):
        if not self.validgauge:
            return -1
        if self.condensed:
            Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi,
                     np.delete(self.MF, self.toignore, axis=0),
                     self.lams, np.delete(self.bigB,self.toignore, axis=0), self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, np.delete(self.mask, self.toignore))

            Eq = MFE_condensed(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
                               self.xi, self.MF,
                               self.lams, self.qmin, self.rhos, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
            return Ep + Eq
        else:
            Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi, self.MF,
            self.lams, self.bigB, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.mask)
        return Ep

    def MFE_alternative(self):
        return self.MFE() - self.lams[0]

    def MFEs(self, chi, chi0, xi, lams, MF, K):

        Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, chi0, xi, MF,
        lams, K, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
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
        if not self.validgauge:
            return -1
        calDispersion(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.theta, self.chi,
                      self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        if show:
            plt.show()

    def minCal(self, K):
        if not self.validgauge:
            return -1
        return minCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.bigB, self.theta,
                      self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def maxCal(self, K):
        if not self.validgauge:
            return -1
        return maxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.bigB, self.theta,
                      self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def minMaxCal(self, K):
        if not self.validgauge:
            return -1
        return minMaxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.bigB, self.theta,
                         self.chi, self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def EMAX(self):
        if not self.validgauge:
            return -1
        return np.sqrt(2 * self.Jzz * EMAX(self.MF, self.lams))

    def TWOSPINON_GAP(self, k):
        if not self.validgauge:
            return -1
        return np.min(self.minCal(k))

    def TWOSPINON_MAX(self, k):
        if not self.validgauge:
            return -1
        return np.max(self.maxCal(k))

    def TWOSPINON_DOMAIN(self, k):
        if not self.validgauge:
            return -1
        A = self.minMaxCal(k)
        return np.min(A[:,0]), np.max(A[:,1])

    def graph_loweredge(self, show):
        if not self.validgauge:
            return -1
        loweredge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.bigB, self.theta, self.chi,
                  self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        if show:
            plt.show()

    def graph_upperedge(self, show):
        if not self.validgauge:
            return -1
        upperedge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.eta, self.h, self.n, self.bigB, self.theta, self.chi,
                  self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        if show:
            plt.show()

    def green_pi(self, k, lam=np.zeros(2)):
        if not self.validgauge:
            return -1
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi(E, V, self.Jzz)

    def green_pi_branch(self, k, lam=np.zeros(2)):
        if not self.validgauge:
            return -1
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi_branch(E, V, self.Jzz), E
    def print_rho(self):
        minLams = self.findminLam(self.chi, self.chi0, self.xi)
        E, V = np.linalg.eigh(self.MF)
        a = min(E[:,0])
        print(a, a+ minLams, self.lams)
        T = np.linspace(minLams-1e-10, minLams+1e-10, 50)
        rho = np.zeros(50)
        for i in range(50):
            rho[i] = rho_true(self.Jzz,self.MF, self.Mcondensed, self.mask, self.dV, T[i]*np.ones(2))[0]
        plt.plot(T, rho)
    def magnetization(self):
        if not self.validgauge:
            return -1
        green = self.green_pi(self.bigB)
        ffact = contract('ik, jk->ij', self.bigB, NN)
        ffactp = np.exp(-1j * ffact)
        ffactm = np.exp(1j * ffact)

        magp = contract('ij, ika, kj, jka->i', ffactp, green[:, 0:4, 4:8], np.exp(1j * self.A_pi_here),
                        piunitcell) / 4
        magm = contract('ij, iak, kj, jka->i', ffactm, green[:, 4:8, 0:4], np.exp(-1j * self.A_pi_here),
                        piunitcell) / 4

        con = 0
        if self.condensed:
            ffact = contract('ik, jk->j', self.qmin, NN)
            ffactp = np.exp(-1j * ffact)
            ffactm = np.exp(1j * ffact)

            tempp = contract('j, k, a, kj, jka->j', ffactp, self.rhos[0:4], self.rhos[4:8], np.exp(1j * self.A_pi_here),
                            piunitcell) / 4
            tempm = contract('j, a, k, kj, jka->j', ffactm, self.rhos[4:8], self.rhos[0:4], np.exp(-1j * self.A_pi_here),
                            piunitcell) / 4

            con = np.mean(tempp+tempm)

        return np.real(np.mean(magp + magm)+con) / 4