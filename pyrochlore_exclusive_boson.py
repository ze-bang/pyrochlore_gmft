import matplotlib.pyplot as plt
import warnings

import numpy as np

from misc_helper import *
from flux_stuff import *
#region Hamiltonian Construction
def M_pi_mag_sub_AB(k, h, n, theta, A_pi_here):
    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    M = contract('ku, u, ru, urx->krx', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                 np.exp(1j*A_pi_here), piunitcell)
    return M


def M_pi_sub_intrahopping_dd(k, alpha, Jpm, A_pi_rs_traced_here):
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * neta(alpha) * ffact)
    M = contract('jl,kjl,ijl, jka, lkb->iab', notrace, -Jpm * A_pi_rs_traced_here / 4, ffact, piunitcell,
                 piunitcell)
    return M


def M_pi(k, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])
    k = contract('ij,jk->ik', k, BasisBZA)

    MAk = M_pi_sub_intrahopping_dd(k, 0, Jpm, A_pi_rs_traced_here)
    MBk = M_pi_sub_intrahopping_dd(k, 1, Jpm, A_pi_rs_traced_here)
    MAnk = M_pi_sub_intrahopping_dd(-k, 0, Jpm, A_pi_rs_traced_here)
    MBnk = M_pi_sub_intrahopping_dd(-k, 1, Jpm, A_pi_rs_traced_here)
    dumz = np.zeros((len(k),4,4))

    MagAkBk = M_pi_mag_sub_AB(k, h, n, theta, A_pi_here)
    MagBkAk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))
    MagAnkBnk = M_pi_mag_sub_AB(-k, h, n, theta, A_pi_here)
    MagBnkAnk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))

    # [dTkAdkA, dTkAbTkA, dTkAdT-kA, dTkAb-kA, dTkAdkB, dTkAbTkB, dTkAdT-kB, dTkAb-kB]
    # [bkAdkA, bkAbTkA, bkAdT-kA, bkAb-kA, bkAdkB, bkAbTkB, bkAdT-kB, bkAb-kB]
    # [d-kAdkA, d-kAbTkA, d-kAdT-kA, d-kAb-kA, d-kAdkB, d-kAbTkB, d-kAdT-kB, d-kAb-kB]
    # [bT-kAdkA, bT-kAbTkA, bT-kAdT-kA, bT-kAb-kA, bT-kAdkB, bT-kAbTkB, bT-kAdT-kB, bT-kAb-kB]
    # [dTkBdkA, dTkBbTkA, dTkBdT-kA, dTkBb-kA, dTkBdkB, dTkBbTkB, dTkBdT-kB, dTkBb-kB]
    # [bkBdkA, bkBbTkA, bkBdT-kA, bkBb-kA, bkBdkB, bkBbTkB, bkBdT-kB, bkBb-kB]
    # [d-kBdkA, d-kBbTkA, d-kBdT-kA, d-kBb-kA, d-kBdkB, d-kBbTkB, d-kBdT-kB, d-kBb-kB]
    # [bT-kBdkA, bT-kBbTkA, bT-kBdT-kA, bT-kBb-kA, bT-kBdkB, bT-kBbTkB, bT-kBdT-kB, bT-kBb-kB]
    MAA = np.block([[MAk, dumz, dumz, MAk],
                   [dumz, MAnk, MAnk, dumz],
                   [dumz, MAnk, MAnk, dumz],
                   [MAk, dumz, dumz, MAk]])
    MAB = np.block([[MagAkBk, dumz, dumz, MagAkBk],
                   [dumz, MagAnkBnk, MagAnkBnk, dumz],
                   [dumz, MagAnkBnk, MagAnkBnk, dumz],
                   [MagAkBk, dumz, dumz, MagAkBk]])
    MBA = np.block([[MagBkAk, dumz, dumz, MagBkAk],
                   [dumz, MagBnkAnk, MagBnkAnk, dumz],
                   [dumz, MagBnkAnk, MagBnkAnk, dumz],
                   [MagBkAk, dumz, dumz, MagBkAk]])
    MBB = np.block([[MBk, dumz, dumz, MBk],
                   [dumz, MBnk, MBnk, dumz],
                   [dumz, MBnk, MBnk, dumz],
                   [MBk, dumz, dumz, MBk]])


    FM = np.block([[MAA, MAB],
                   [MBA, MBB]])
    FM = FM + np.diag(np.ones(32)*Jzz/2)
    return FM


def bogoliubov(M):
    J = np.diag(np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]))
    dJ = np.diag(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]))
    Lower = np.linalg.cholesky(M)
    Upper = np.transpose(np.conj(Lower),(0,2,1))
    ToD = contract('iab,bc,icd->iad',Upper,J,Lower)
    L, U = np.linalg.eigh(ToD)
    U = np.flip(U, axis=2)
    L = np.flip(L, axis=1)
    dum = np.identity(M.shape[1])
    L = contract('ia,ab->iab',L,dum)
    E = contract('ab,ibc->iac',dJ,L)
    P = contract('iab, ibc, icd->iad',np.linalg.inv(Upper),U,np.sqrt(E))
    # test1 = contract('iab, ibc, icd -> iad', np.transpose(np.conj(P),(0,2,1)), M, P)
    # test2 = contract('iab, bc, icd -> iad', np.transpose(np.conj(P),(0,2,1)), J, P)
    return np.diagonal(E, 0,1,2), P

def bose_einstein(omega,T):
    return 1/(np.exp(omega/T)-1)

class piFluxSolver:
    def __init__(self, Jxx, Jyy, Jzz, theta=0, h=0, n=np.array([0, 0, 0]), kappa=2, lam=2, BZres=20, graphres=20,
                 ns=1, tol=1e-10, flux=np.zeros(4), intmethod=riemann_sum_3d_pts, T=1):
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


        self.A_pi_here, self.equi_class_field, self.equi_class_flux, self.gen_equi_class_field, self.gen_equi_class_flux = determineEquivalence(n, flux)


        self.pts, self.weights = self.intmethod(0, 1, 0, 1, 0, 1, BZres)

        self.minLams = np.zeros(2, dtype=np.double)

        self.BZres = BZres
        self.graphres = graphres

        self.toignore = np.array([],dtype=int)
        self.q = np.nan
        self.qmin = np.empty(3)
        self.qmin[:] = np.nan
        self.qminB = np.copy(self.qmin)
        self.qminT = np.copy(self.qmin)
        self.condensed = False
        self.delta = np.zeros(16)
        self.rhos = np.zeros(16)

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

        self.MF = M_pi(self.pts, self.Jzz, self.Jpm,self.Jpmpm,self.h,self.n,self.theta,self.chi,self.chi0,self.xi,self.A_pi_here,self.A_pi_rs_traced_here,self.A_pi_rs_traced_pp_here)
        self.E, self.V = bogoliubov(self.MF)
        self.omega = bose_einstein(self.E, T)

    def M_pi(self, k):
        return  M_pi(k, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n,
                     self.theta, self.chi, self.chi0, self.xi, self.A_pi_here
                     , self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)


    def occu_num(self):
        P11 = self.V[:,0:8,0:8]
        P12 = self.V[:,8:16,8:16]
        nbb = contract('ijk,ijk,ik->ij', np.conj(P11)[:,0:4,:], P11[:,0:4,:], self.omega[:,0:8]) + \
              contract('ijk,ijk,ik->ij', np.conj(P12)[:,0:4,:], P12[:,0:4,:], self.omega[:,8:16])
        ndd = contract('ijk,ijk,ik->ij', np.conj(P11)[:,4:8,:], P11[:,4:8,:], self.omega[:,0:8]) + \
              contract('ijk,ijk,ik->ij', np.conj(P12)[:,4:8,:], P12[:,4:8,:], self.omega[:,8:16])
        return np.mean(nbb+ndd)