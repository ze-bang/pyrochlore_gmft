import matplotlib.pyplot as plt
import warnings

import numpy as np
from core.misc_helper import *
from core.flux_stuff import *
import scipy as sp


#region Hamiltonian Construction
def M_pi_mag_sub_AB(k, h, n, theta, A_pi_here):
    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(-1j * ffact)
    M = contract('ku, u, ru, urx->krx', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                 np.exp(1j*A_pi_here), piunitcell)
    return M
def M_pi_sub_intrahopping_dd(k, alpha, Jpm, A_pi_rs_traced_here):
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * neta(alpha) * ffact)
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
    # MAnk = M_pi_sub_intrahopping_dd(-k, 0, Jpm, A_pi_rs_traced_here)
    # MBnk = M_pi_sub_intrahopping_dd(-k, 1, Jpm, A_pi_rs_traced_here)

    MagAkBk = M_pi_mag_sub_AB(k, h, n, theta, A_pi_here)
    MagBkAk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))
    # MagAnkBnk = M_pi_mag_sub_AB(-k, h, n, theta, A_pi_here)
    # MagBnkAnk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))

    KK = np.block([[MAk, MagAkBk],
                   [MagBkAk, MBk]])

    # nKnK = np.block([[MAnk, MagAnkBnk],
    #                [MagBnkAnk, MBnk]])
    dumz = np.zeros((len(k),8,8))

    FM = np.block([[KK, dumz, dumz, KK],
                   [dumz, KK, KK, dumz],
                   [dumz, KK, KK, dumz],
                   [KK, dumz, dumz, KK]])
    FM = FM + np.diag(np.ones(32)*Jzz/2)
    return FM

def bogoliubov(M):
    dim = int(M.shape[1]/2)
    J = np.concatenate((np.ones(dim),-np.ones(dim)))
    J = np.diag(J)
    Lower = np.linalg.cholesky(M)
    Upper = np.transpose(np.conj(Lower),(0,2,1))
    ToD = contract('iab,bc,icd->iad',Upper,J,Lower)
    L, U = np.linalg.eigh(ToD)
    U = np.flip(U, axis=2)
    L = np.flip(L, axis=1)
    dum = np.identity(M.shape[1])
    L = contract('ia,ab->iab',L,dum)
    E = contract('ab,ibc->iac',J,L)
    P = contract('iab, ibc, icd->iad',np.linalg.inv(Upper),U,np.sqrt(E))
    E = np.diagonal(E, 0,1,2)
    return E, P


#region initialize tensor matrix to multiply through

# A = np.array([[1,0,0,1],
#               [0,1,1,0],
#               [0,1,1,0],
#               [1,0,0,1]])
# A = A + 1/2*np.identity(4)
# #
# Ep, Vp = ldl_single(A)
# print(Ep, Vp)
#endregion



def M_pi_smol(k, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    chi = chi * np.array([chi_A, chi_A])
    chi0 = chi0 * np.ones((2, 4))
    xi = xi * np.array([xipicell[0], xipicell[0]])
    k = contract('ij,jk->ik', k, BasisBZA)

    MAk = M_pi_sub_intrahopping_dd(k, 0, Jpm, A_pi_rs_traced_here)
    MBk = M_pi_sub_intrahopping_dd(k, 1, Jpm, A_pi_rs_traced_here)

    MagAkBk = M_pi_mag_sub_AB(k, h, n, theta, A_pi_here)
    MagBkAk = np.conj(np.transpose(MagAkBk, (0, 2, 1)))

    KK = np.block([[MAk, MagAkBk],
                   [MagBkAk, MBk]])
    return KK





def E_pi(k, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    M = M_pi(k, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
    E, V = bogoliubov(M)
    return E, V


def dispersion_pi(lams, k, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here):
    temp = E_pi(k, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, chi0, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)[0]
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


def bose_einstein(omega,T):
    return 1/(np.exp(omega/T)-1)

class piFluxSolver:
    def __init__(self, Jxx, Jyy, Jzz, theta=0, h=0, n=np.array([0, 0, 0]), kappa=2, lam=2, BZres=20, graphres=20,
                 ns=1, tol=1e-10, flux=np.zeros(4), intmethod=trapezoidal_rule_3d_pts, T=1, gzz=2.24, Breal=False):
        self.intmethod = intmethod
        J = np.array([Jxx, Jyy, Jzz])
        a = np.argmax(J)
        xx = np.mod(a-2,3)
        yy = np.mod(a-1,3)
        self.Jzz = J[a]
        self.Jpm = -(J[xx] + J[yy]) / 4
        self.Jpmpm = (J[xx] - J[yy]) / 4
        self.theta = theta
        self.kappa = kappa
        self.tol = tol
        self.lams = np.array([lam, lam], dtype=np.double)
        self.ns = ns
        if Breal:
            self.h = 5.7883818060*10**(-2)*h*gzz
        else:
            self.h = h
        if a == 0:
            self.h = -1j*self.h
        self.n = n
        self.chi = 0.18
        self.xi = 0.5
        self.chi0 = 0.18

        self.flux = flux
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
        # self.Green = bose_einstein(self.E, T)

    def M_pi(self, k):
        return M_pi(k, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n,
                     self.theta, self.chi, self.chi0, self.xi, self.A_pi_here
                     , self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def solvemeanfield(self, tol=1e-13):
        num_last = 0
        num_now = self.occu_num()
        while np.abs(num_last-num_now)>tol:
            # print(num_now, num_last, np.abs(num_last-num_now))
            self.MF = M_pi(self.pts, self.Jzz, self.Jpm/(1+num_now),self.Jpmpm,self.h/(1+num_now),self.n,self.theta,self.chi,self.chi0,self.xi,self.A_pi_here,self.A_pi_rs_traced_here,self.A_pi_rs_traced_pp_here)
            self.E, self.V = bogoliubov(self.MF)
            num_last = num_now
            num_now = self.occu_num()

    def occu_num(self):
        P11 = self.V[:,0:16,0:16]
        P12 = self.V[:,0:16,16:32]
        ndd = contract('ijk,ijk->ij', np.conj(P11)[:,0:8,:], P11[:,0:8,:]) + \
              contract('ijk,ijk->ij', np.conj(P12)[:,0:8,:], P12[:,0:8,:])
        nbb = contract('ijk,ijk->ij', np.conj(P11)[:,8:16,:], P11[:,8:16,:]) + \
              contract('ijk,ijk->ij', np.conj(P12)[:,8:16,:], P12[:,8:16:])
        ND = np.mean(ndd,axis=1)
        NB = np.mean(nbb,axis=1)
        N = np.dot(NB+ND,self.weights)
        return np.real(N - 2)

    def graph(self, show):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi,
                      self.chi0, self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        if show:
            plt.show()

    def GS(self):
        return (contract('i,i->',np.mean(self.E,axis=1),self.weights) - 0.5)

    def GS_gauge_correction(self):
        return self.GS()+HanYan_GS(self.Jpm,self.Jzz,self.h,self.n,self.flux)