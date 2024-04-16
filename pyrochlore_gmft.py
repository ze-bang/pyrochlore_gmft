import matplotlib.pyplot as plt
import warnings
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
    tempchi0 = chi[:,0,0]
    M1 = contract('jl, kjl, kjl, i, km->ikm', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, chi, d, di)
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

        FM = np.block([[MAk, MagAkBk, MAdkAdnk, dummy],
                       [MagBkAk, MBk, dummy, MBdkBdnk],
                       [MAnkAk, dummy, MAnk, MagAnkBnk],
                       [dummy, MBnkBk, MagBnkAnk, MBnk]])

    return FM

#endregion

#region E_pi
def E_pi_fixed(lams, M):
    # M = M + np.diag(np.repeat(np.repeat(lams, int(M.shape[1]/4)), 2))
    M = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
    E, V = np.linalg.eigh(M)
    return [E, V]


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
        return 1/(2*kappa**2), np.array([0,0,0]).reshape((1,3)), np.array([0,0,0]).reshape((1,3))

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
    lamMax = 10*np.ones(2)
    lams = lamMax
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
            if (abs(lamlast - lams) < 1e-15).all() or ((np.absolute(rhoguess - kappa) <= tol).all()):
                break
        except:
            lamMin = lams
    warnings.resetwarnings()
    return lams

#endregion

#region Mean field calculation

def chi_integrand(k, E, V, Jzz):
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jlk->ijl', k, NNminus)
    ffactB = np.exp(-1j * ffact)
    A = contract('iab, ijl,jka, lkb->kjil', green[:, 12:16, 4:8], ffactB, piunitcell, piunitcell)
    return A

def chiCal(E, V, Jzz, n, n1, n2, n3, n4, n5, pts, weights, unitcellCoord):
    M = integrate_fixed(chi_integrand, weights, pts, E, V, Jzz)
    M1 = chi_mean_field(n, M[0], n1, n2, n3, n4, n5, unitcellCoord)
    return M1

def xi_integrand(k, E, V, Jzz):
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jk->ij', k, NN)
    ffactA = np.exp(1j * ffact)
    A = contract('ika, ij,jka->kij', green[:, 0:4, 4:8], ffactA, piunitcell)
    return A
def xiCal(E, V, Jzz, n, n1, n2, n4, n5, pts, weights, unitcellCoord):
    M = integrate_fixed(xi_integrand, weights, pts, E, V, Jzz)
    M1 = xi_mean_field(n, M, n1, n2, n4, n5, unitcellCoord)
    return M1

def calmeanfield(E, V, Jzz, n, n1, n2, n3, n4, n5, pts, weights, unitcellCoord=piunitcellCoord):
    chi = chiCal(E, V, Jzz, n, n1, n2, n3, n4, n5, pts, weights, unitcellCoord)
    return chi, xiCal(E, V, Jzz, n, n1, n2, n4, n5, pts, weights, unitcellCoord)

# endregion

#region graphing BZ

def dispersion_pi(lams, k, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, unitcell):
    temp = np.sqrt(2 * Jzz * E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                  A_pi_rs_traced_pp_here, unitcell)[0])
    return temp


def calDispersion(lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
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
    dUW = dispersion_pi(lams, UW, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                        A_pi_rs_traced_pp_here, unitcell)

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
        # print(temp[i],i)
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
    dUW = minCal(lams, UW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)

    plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, 'b',zorder=10)
    plt.plot(np.linspace(gX, gW1, len(dXW)), dXW, 'b',zorder=10)
    plt.plot(np.linspace(gW1, gK, len(dWK)), dWK, 'b',zorder=10)
    plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, 'b',zorder=10)
    plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, 'b',zorder=10)
    plt.plot(np.linspace(gL, gU, len(dLU)), dLU, 'b',zorder=10)
    plt.plot(np.linspace(gU, gW2, len(dUW)), dUW, 'b',zorder=10)

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
    return min(np.min(dGammaX), np.min(dXW), np.min(dWK),np.min(dKGamma),np.min(dGammaL),np.min(dLU),np.min(dUW))

def upperedge(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
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
    dUW = maxCal(lams, UW, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                 A_pi_rs_traced_pp_here, unitcell)

    plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, 'b',zorder=10)
    plt.plot(np.linspace(gX, gW1, len(dXW)), dXW, 'b',zorder=10)
    plt.plot(np.linspace(gW1, gK, len(dWK)), dWK, 'b',zorder=10)
    plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, 'b',zorder=10)
    plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, 'b',zorder=10)
    plt.plot(np.linspace(gL, gU, len(dLU)), dLU, 'b',zorder=10)
    plt.plot(np.linspace(gU, gW2, len(dUW)), dUW, 'b',zorder=10)

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
    return max(np.max(dGammaX), np.max(dXW), np.max(dWK),np.max(dKGamma),np.max(dGammaL),np.max(dLU),np.max(dUW))
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

def graphing_M_setup(flux):
    if (flux == np.zeros(4)).all():
        unitCellgraph = np.array([[[1]],[[1]],[[1]],[[1]]])
        A_pi_here = np.array([[0,0,0,0]])
        unitcellCoord = np.array([[0,0,0]])
    elif (flux == np.pi*np.ones(4)).all():
        unitCellgraph = piunitcell
        A_pi_here = A_pi
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

    elif (flux == np.array([np.pi,0,0, np.pi])).all():
        unitCellgraph = np.array([[[1,0],
                                    [0,1]],
                                    [[1,0],
                                    [0,1]],
                                    [[0,1],
                                    [1,0]],
                                    [[0,1],
                                    [1,0]]
                            ])
        A_pi_here = np.array([[0,0,0,0],
                                [0,np.pi,0,0]])
        unitcellCoord = np.array([[0, 0, 0],[0,1,1]])

    elif (flux == np.array([0, np.pi, np.pi, 0])).all():
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
                                [0,0,np.pi,0]])
        unitcellCoord = np.array([[0, 0, 0],[0,1,0]])
    return unitCellgraph, A_pi_here, unitcellCoord

#endregion
class piFluxSolver:
    def __init__(self, Jxx, Jyy, Jzz, theta=0, h=0, n=h110, kappa=2, lam=2, BZres=20, graphres=20,
                 ns=1, tol=1e-10, flux=np.zeros(4), intmethod=gauss_quadrature_3D_pts, gzz=2.24, Breal=False):
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
        self.flux = flux
        self.A_pi_here, self.n1, self.n2, self.equi_class_field, self.equi_class_flux, self.gen_equi_class_field, self.gen_equi_class_flux = determineEquivalence(n, flux)
        self.n3 = self.n4 = self.n5= 0
        self.pts, self.weights = self.intmethod(0, 1, 0, 1, 0, 1, BZres)

        self.xi = xi_mean_field(n, 0.002*np.random.rand()*np.ones((4,4)),self.n1,self.n2,self.n4,self.n5,piunitcellCoord)
        self.chi = chi_mean_field(n, 0.002*np.random.rand()*np.ones((4,4)),self.n1,self.n2,self.n3,self.n4,self.n5,piunitcellCoord)

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

        self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.A_pi_rs_rsp_here, self.A_pi_rs_rsp_pp_here = gen_gauge_configurations(self.A_pi_here)
        self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                       self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        self.E, self.V = np.linalg.eigh(self.MF)

    def findLambda(self):
        return findlambda_pi(self.kappa, self.tol,self.minLams, self.Jzz, self.weights, self.E)

    def findLambda_unconstrained(self):
        return findlambda_pi(self.kappa,self.tol, np.zeros(2), self.Jzz, self.weights, self.E)


    def findminLam(self):
        searchGrid=34
        B = genBZ(searchGrid)
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(A_pi_here)
        xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        M = M_pi(B, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi, A_pi_here,
                 A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)
        minLams, self.qmin = findminLam_scipy(M, B, self.tol, self.Jpm, self.Jpmpm, self.h, self.n,
                                        self.theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph,
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
        chi, xi = calmeanfield(E, V, self.Jzz, self.n, self.n1, self.n2, self.n3, self.n4, self.n5, self.pts, self.weights)
        return chi, xi

    def solvemufield(self):
        self.findminLam()
        self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                       self.A_pi_here,
                       self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        self.E, self.V = np.linalg.eigh(self.MF)
        self.lams = self.findLambda()
        return self.GS()

    def solvemeanfield(self, tol=1e-10):
        if self.Jpmpm == 0:
            self.chi = np.zeros((4,4,4))
            self.xi = np.zeros((4,4))
            self.condensation_check()
        else:
            self.findminLam()
            self.lams = self.findLambda()
            self.chi, self.xi = self.calmeanfield()
            GS = self.solvemufield()
            count = 0
            while True:
                chilast, xilast, GSlast = np.copy(self.chi), np.copy(self.xi), GS
                chi, xi = self.calmeanfield()
                self.chi, self.xi = chi, xi
                GS = self.solvemufield()
                # print(self.chi[0], self.xi[0,0], self.GS())
                count = count + 1
                if ((abs(self.chi-chilast) < tol).all() and (abs(self.xi-xilast) < tol).all()) or count >=100:
                    break
            self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                           self.A_pi_here,
                           self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
            self.E, self.V = np.linalg.eigh(self.MF)
            self.condensation_check()
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
        self.condensed = A < (deltamin /(self.BZres**3)) ** 2

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
        self.lams = self.findLambda()
        self.set_condensed()
        self.ifcondense()
        self.set_delta()


    def M_true(self, k):
        return M_pi(k, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                    self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def E_pi_mean(self, k):
        return np.mean(np.sqrt(2 * self.Jzz *
                               E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                                    self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)[0]), axis=1)

    def E_pi(self, k):
        return np.sqrt(2 * self.Jzz *
                       E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                            self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)[0])


    def dispersion(self, k):
        return dispersion_pi(self.lams, k, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi,
                             self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def LV_zero(self, k, lam=np.zeros(2)):
        if np.any(lam == 0):
            lam = self.lams
        return E_pi(k, lam, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                    self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)

    def GS(self):
        return integrate(self.E_pi_mean, self.pts, self.weights) - self.kappa*self.lams[0]


    def MFE(self):
        # if self.condensed:
        # Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi,
        #          self.lams, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.BZres, self.kappa, np.delete(self.pts,self.toignore), np.delete(self.weights,self.toignore))
        Ep = self.GS()
        # Eq = MFE_condensed(self.qminB, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0,
        #                    self.xi, self.lams, self.rhos, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        # print(Ep, Eq)
        return Ep
        # else:
        # Ep = MFE(self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.chi0, self.xi,
        # self.lams, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.BZres, self.kappa, self.pts, self.weights)
        # return Ep
    def graph_raw(self, show):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                      self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here)
        if show:
            plt.show()

    def graph(self, show):
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(A_pi_here)
        xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        calDispersion(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi,
                      A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)
        if show:
            plt.show()

    def minCal(self, K):
        return minCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, self.chi,
                      self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitcell)

    def maxCal(self, K):
        return maxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, self.chi,
                      self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitcell)

    def minMaxCal(self, K):
        return minMaxCal(self.lams, K, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, self.chi,
                         self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.unitcell)

    def EMAX(self):
        return np.sqrt(2 * self.Jzz * EMAX(self.MF, self.lams))

    def TWOSPINON_GAP(self, k):
        return np.min(self.minCal(k))

    def TWOSPINON_MAX(self, k):
        return np.max(self.maxCal(k))

    def TWOSPINON_DOMAIN(self, k):
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(A_pi_here)
        xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        q = np.sqrt(2 * self.Jzz *
                       E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi,
                            A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)[0])
        mindex = np.argmin(q[:,0])
        maxdex = np.argmax(q[:,-1])
        kmin = k[mindex].reshape((1,3))
        kmax = k[maxdex].reshape((1,3))
        A = DSSF_E_DOMAIN(self.lams, kmin, kmax, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta,
                          chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here)
        print(A, kmin, kmax)
        return A

    def graph_loweredge(self, show):
        unitCellgraph, A_pi_here,unitcellCoord = graphing_M_setup(self.flux)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(
            A_pi_here)
        xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        min = loweredge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, chi, xi,
                  A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)
        if show:
            plt.show()
        return min

    def graph_upperedge(self, show):
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(
            A_pi_here)
        xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        max = upperedge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.pts, self.theta, chi, xi,
                  A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)
        if show:
            plt.show()
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
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(
            A_pi_here)
        xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        E, V = E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi, A_pi_here,
             A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi(E, V, self.Jzz)

    def green_pi_branch_reduced(self, k):
        unitCellgraph, A_pi_here, unitcellCoord = graphing_M_setup(self.flux)
        A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here = gen_gauge_configurations(
            A_pi_here)
        xi = xi_mean_field(self.n, self.xi, self.n1, self.n2, self.n4, self.n5, unitcellCoord)
        chi = chi_mean_field(self.n, self.chi[0], self.n1, self.n2, self.n3, self.n4, self.n5, unitcellCoord)
        E, V = E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, chi, xi, A_pi_here,
             A_pi_rs_traced_here, A_pi_rs_traced_pp_here, unitCellgraph)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi_branch(E, V, self.Jzz), E, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here, unitCellgraph

    def mag_integrand(self, k):
        M = M_pi(k, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                 self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here) + np.diag(np.repeat(np.repeat(self.lams, 4), 2))
        E, V = np.linalg.eigh(M)
        E = np.sqrt(2 * self.Jzz * E)
        green = green_pi(E, V, self.Jzz)

        ffact = contract('ik, jk->ij', k, NN)
        ffact = np.exp(1j * ffact)

        magp = np.real(contract('ku, ru, krx, urx->k', ffact,
                             np.exp(1j * self.A_pi_here), green[:, 0:4, 4:8], piunitcell))
        return magp


    def magnetization(self):
        mag = np.abs(integrate(self.mag_integrand, self.pts, self.weights))/16
        con = 0
        if self.condensed:
            ffact = contract('ik, jk->ij', self.qminB, NN)
            ffactp = np.exp(1j * ffact)
            ffactm = np.exp(-1j * ffact)

            tempp = contract('ij, k, a, kj, jka->i', ffactp, self.rhos[0:4], self.rhos[4:8], np.exp(1j * self.A_pi_here),
                            piunitcell) / 4
            tempm = contract('ij, a, k, kj, jka->i', ffactm, self.rhos[4:8], self.rhos[0:4], np.exp(-1j * self.A_pi_here),
                            piunitcell) / 4
            con = np.mean(tempp+tempm)

        return  np.real(mag+con)