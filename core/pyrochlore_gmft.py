"""Gauge Mean Field Theory (GMFT) solver for the pyrochlore lattice.

Implements the spinon mean-field decoupling of the nearest-neighbour XXZ spin
model on the pyrochlore lattice with a pi-flux ansatz.  The spin-1/2 operators
are rewritten in terms of bosonic spinon creation/annihilation operators using
the Schwinger boson representation.  A Z2 gauge structure (IGG) emerges from
the redundancy of the parton construction, and the physical gauge fluxes
through the elementary hexagonal plaquettes label distinct quantum spin-liquid
ansaetze (0-flux, pi-flux, etc.).

The mean-field Hamiltonian is a quadratic bosonic form:
    H_MF = sum_k  Psi_k^dag  M(k)  Psi_k
where Psi_k = (a_{k,s}, b_{k,s}, a^dag_{-k,s}, b^dag_{-k,s}) is the Nambu
spinor of the A/B sublattice spinon operators, and M(k) contains hopping
(Jpm), pairing (Jpmpm), Zeeman (h), and fictitious Z2 (g) terms.  The
Lagrange multipliers (lambda) enforce the single-occupancy constraint
<n_i> = kappa on every site.

Self-consistency loop:
    1.  Diagonalise H_MF  -->  spinon band structure E(k)
    2.  Determine lambda via bisection so that the constraint is satisfied.
    3.  Compute the mean-field order parameters  xi  (hopping) and  chi
        (pairing) from the spinon Green's functions.
    4.  Reconstruct M(k) with the new xi, chi and repeat until convergence.
    5.  Check for Bose-Einstein condensation of the lowest spinon mode.

Physical observables (magnetisation, dynamic structure factor edges, gap)
are extracted from the converged solution.

Key references:
    - Savary & Balents, Rep. Prog. Phys. 80, 016502 (2017)
    - Lee, Onoda & Balents, Phys. Rev. B 86, 104412 (2012)

Notation
--------
- A_pi : gauge connection (U(1) phases) on the bonds of the enlarged unit cell
- NN   : nearest-neighbour displacement vectors (sublattice positions)
- NNminus : pairwise differences NN[i]-NN[j] (used in intra-sublattice terms)
- z    : local [111] axes for the four sublattices
- notrace : 4x4 matrix with 1's everywhere except the diagonal (excludes self-loops)
- piunitcell : unitary matrices encoding the sublattice embedding in the
  enlarged magnetic unit cell for the pi-flux state
- BasisBZA : reciprocal-lattice vectors of the FCC Brillouin zone
- contract : opt_einsum.contract  (Einstein summation)
"""

import matplotlib.pyplot as plt
import warnings

import numpy as np

from core.misc_helper import *
from core.flux_stuff import *
import time
from scipy.optimize import minimize

#region Hamiltonian Construction

def M_pi_mag_sub_AB(k, h, n, theta, A_pi_here, unitcell=piunitcell):
    """Zeeman (magnetic field) coupling block between A and B sublattices.

    Computes the off-diagonal block of the spinon Hamiltonian that couples
    the A-sublattice spinon operators to the B-sublattice ones via the
    external magnetic field h along direction n.  The phase theta parametrises
    a U(1) rotation of the field in the local frame.  The gauge connection
    A_pi_here enters through the Peierls-like phase factors exp(i A_rs).

    Parameters
    ----------
    k : ndarray, shape (nk, 3)
        Momenta in Cartesian reciprocal-space coordinates.
    h : float
        External magnetic field strength.
    n : ndarray, shape (3,)
        Unit vector giving the field direction (e.g. [1,1,0]/sqrt(2)).
    theta : float
        Angle parametrising the in-plane field rotation.
    A_pi_here : ndarray, shape (ncell, 4)
        Gauge connection for each unit cell and sublattice.
    unitcell : ndarray
        Sublattice embedding matrices.

    Returns
    -------
    M : ndarray, shape (nk, size, size)
        The A-B Zeeman block of the Hamiltonian for each k-point.
    """
    zmag = contract('k,ik->i', n, z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    M = contract('ku, u, ru, urx->krx', -1 / 4 * h * ffact * (np.cos(theta) - 1j * np.sin(theta)), zmag,
                 np.exp(1j*A_pi_here), unitcell)
    return M

def M_pi_sub_intrahopping_AA(k, alpha, Jpm, A_pi_rs_traced_here, unitcell=piunitcell):
    """Intra-sublattice (A-A or B-B) hopping block of the spinon Hamiltonian.

    This is the diagonal block arising from the transverse exchange Jpm = -(Jxx+Jyy)/4.
    alpha=0 gives the A-A block; alpha=1 gives the B-B block (the two differ
    by a sign in the Fourier phase convention, controlled by neta(alpha) = +/-1).

    The gauge phases A_pi_rs_traced_here = exp[i(A_rs - A_rs')] encode the
    background Z2 flux through the plaquettes.
    """
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * neta(alpha) * ffact)
    M = contract('jl,kjl,ijl, jka, lkb->iab', notrace, -Jpm * A_pi_rs_traced_here / 4, ffact, unitcell,
                 unitcell)
    return M


def M_pi_sub_interhopping_AB(k, Jpmpm, xi, A_pi_rs_traced_pp_here, unitcell=piunitcell):
    """Inter-sublattice A-B hopping block from the bond-anisotropic exchange Jpmpm.

    Jpmpm = (Jxx - Jyy)/4 is the XY-anisotropy coupling.  The mean-field
    decoupling introduces the hopping order parameter xi_{rs} = <a^dag_r b_s>,
    which enters both as xi and conj(xi).  The gauge phases
    A_pi_rs_traced_pp_here = exp[i(A_rs + A_rs')] account for the doubled
    gauge connection in the anomalous channel.
    """
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j * ffact)
    M1a = contract('jl, kjl, ij, kl, jkx->ikx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, xi, unitcell)
    M1b = contract('jl, kjl, il, kj, lkx->ikx', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, xi, unitcell)

    M2a = contract('jl, kjl, ij, kl, jkx->ixk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(xi), unitcell)
    M2b = contract('jl, kjl, il, kj, lkx->ixk', notrace, Jpmpm / 4 * A_pi_rs_traced_pp_here, ffact, np.conj(xi), unitcell)
    return M1a + M1b + M2a + M2b


def M_pi_sub_pairing_AdAd(k, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell=piunitcell):
    """Anomalous (pairing) block a^dag a^dag for A-sublattice spinons.

    This Bogoliubov pairing term arises from the Jpmpm exchange and couples
    spinon creation operators at k and -k.  The pairing amplitude is
    proportional to the mean-field parameter chi (the anomalous Green's
    function <a_r a_s>).  chi[0] and chi[1] correspond to the two sublattice
    sectors.
    """
    d = np.ones(len(k))
    di = np.identity(unitcell.shape[1])
    M1 = contract('jl, kjl, kjl, i, km->ikm', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, chi[1], d, di)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    tempchi0 = chi[1, :,0,0]
    M2 = contract('jl, kjl, ijl, k, jka, lkb->iba', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0, unitcell,
                  unitcell)
    return M1 + M2

def M_pi_sub_pairing_BdBd(k, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell=piunitcell):
    """Anomalous (pairing) block b^dag b^dag for B-sublattice spinons.

    Analogous to M_pi_sub_pairing_AdAd but for the B sublattice.
    """
    d = np.ones(len(k))
    di = np.identity(unitcell.shape[1])
    M1 = contract('jl, kjl, kjl, i, km->ikm', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, chi[0], d, di)

    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(1j * ffact)
    tempchi0 = chi[0, :,0,0]
    M2 = contract('jl, kjl, ijl, k, jka, lkb->iba', notrace, Jpmpm * A_pi_rs_traced_pp_here / 8, ffact, tempchi0, unitcell,
                  unitcell)
    return M1 + M2

def M_pi_sub_pairing_BB(k, Jpmpm, chi, A_pi_rs_traced_pp_here, unitcell=piunitcell):
    """Normal pairing block b b for B-sublattice (Hermitian conjugate partner).

    Completes the Bogoliubov structure together with the b^dag b^dag block.
    """
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(-1j * ffact)
    tempchi0 = np.conj(chi[0, :,0,0])
    M2 = contract('jl, kjl, ijl, k, jka, lkb->iab', notrace, Jpmpm * A_pi_rs_traced_pp_here / 4, ffact, tempchi0, unitcell,
                  unitcell)
    return M2

def M_pi_fictitious_Z2_AA(k, alpha, A_pi_rs_traced_pp_here, g, unitcell=piunitcell):
    """Fictitious Z2 gauge field contribution to the intra-sublattice block.

    The parameter g couples to the Z2 gauge structure of the parton
    construction.  This term can be used to probe the stability of the
    spin-liquid ansatz against Z2 gauge fluctuations.  It acts like an
    additional intra-sublattice hopping modulated by the combined gauge
    phases A_pi_rs_traced_pp.
    """
    ffact = contract('ik, jlk->ijl', k, NNminus)
    ffact = np.exp(neta(alpha)* 1j * ffact)
    M2 = contract('jl, kjl, ijl, jka, lkb->iba', notrace, g * A_pi_rs_traced_pp_here / 4, ffact, unitcell,
                  unitcell)
    return M2

def M_pi(k, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
         unitcell=piunitcell, cartesian=False):
    """Assemble the full spinon mean-field Hamiltonian matrix M(k).

    Depending on whether Jpmpm (XY anisotropy) is zero or not, the matrix
    has a different structure:

    * Jpmpm == 0:  2-block form (no Bogoliubov pairing needed)
        M(k) = [[  A(k)     ,  Mag_AB(k) ],
                [  Mag_BA(k),  B(k)      ]]
      where A(k), B(k) are the intra-sublattice hopping blocks and
      Mag_AB is the Zeeman inter-sublattice coupling.

    * Jpmpm != 0:  4-block Nambu (Bogoliubov-de Gennes) form
        M(k) = [[  A(k)      ,  Mag_AB(k)+Hop_AB ,  Pair_AdAd ,  0         ],
                [  Mag_BA(k)+Hop_BA ,  B(k)       ,  0         ,  Pair_BdBd ],
                [  Pair_AnkAk,  0                 ,  A(-k)     ,  Mag_BnkAnk],
                [  0         ,  Pair_BnkBk        ,  Mag_AnkBnk,  B(-k)     ]]
      This doubles the matrix dimension to account for the particle-hole
      (anomalous) channels.

    Parameters
    ----------
    k : ndarray, shape (nk, 3)
        Momenta in reciprocal-lattice coordinates (converted to Cartesian
        internally unless cartesian=True).
    Jpm : float
        Transverse exchange coupling -(Jxx + Jyy)/4.
    Jpmpm : float
        XY-anisotropy coupling (Jxx - Jyy)/4.
    h : float
        External magnetic field strength.
    n : ndarray, shape (3,)
        Field direction unit vector.
    theta : float
        In-plane field rotation angle.
    chi : ndarray
        Pairing mean-field order parameter (anomalous correlator).
    xi : ndarray
        Hopping mean-field order parameter (normal correlator).
    A_pi_here : ndarray
        Gauge connection on the sublattice bonds.
    A_pi_rs_traced_here : ndarray
        Traced gauge phases exp[i(A_rs - A_rs')].
    A_pi_rs_traced_pp_here : ndarray
        Traced gauge phases exp[i(A_rs + A_rs')].
    g : float
        Fictitious Z2 gauge coupling strength.
    unitcell : ndarray
        Sublattice embedding matrices.
    cartesian : bool
        If True, k is already in Cartesian coordinates; otherwise it is
        converted via BasisBZA.

    Returns
    -------
    FM : ndarray, shape (nk, dim, dim)
        The full Hamiltonian matrix at each k-point.
    """
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

        FM = np.block([[MAk, MagAkBk, MAdkAdnk, dummy],
                       [MagBkAk, MBk, dummy, MBdkBdnk],
                       [MAnkAk, dummy, MAnk, MagBnkAnk],
                       [dummy, MBnkBk, MagAnkBnk, MBnk]])

    return FM

#endregion

#region E_pi

def E_pi_fixed(lams, M):
    """Diagonalise the Hamiltonian M with a fixed Lagrange multiplier.

    Adds lambda * I to the diagonal (enforcing the single-occupancy
    constraint at the mean-field level) and returns the eigenvalues E
    and eigenvectors V.
    """
    M = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
    E, V = np.linalg.eigh(M)
    return E, V


def E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
         unitcell=piunitcell, cartesian=False):
    """Build the full Hamiltonian at momenta k, add the Lagrange multiplier,
    and return the eigenvalues E(k) and eigenvectors V(k).

    The eigenvalues are the *unsquared* spinon energies; the physical
    spinon dispersion is  omega(k) = sqrt(2 * Jzz * E(k)).
    """
    M = M_pi(k, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell, cartesian)
    M = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
    E, V = np.linalg.eigh(M)
    return [E, V]

#endregion

#region find lambda
# The Lagrange multiplier lambda enforces the mean-field constraint
# <n_i> = kappa (average spinon occupation per site).  The functions below
# compute the constraint integrand and solve for lambda via bisection.

def I3_integrand(E, lams, Jzz):
    """Constraint integrand: average spinon density at zero temperature.

    For a bosonic system with dispersion omega_k = sqrt(2*Jzz*(E_k + lam)),
    the ground-state occupation is  <n> = (Jzz / omega_k) summed over bands,
    which must equal kappa.  This function returns the integrand that is
    averaged over the Brillouin zone.
    """
    size = int(E.shape[1]/2)
    E = np.sqrt(2*Jzz*(E+np.repeat(lams, size)))
    Ep = Jzz / E
    return np.mean(Ep,axis=1)

def I3_integrand_site(E, V, lams, Jzz, xyz):
    """Site-resolved constraint integrand.

    Like I3_integrand but projected onto individual sublattice sites using
    the eigenvector weights |V_{k,n,s}|^2, allowing site-dependent lambda.
    The xyz flag indicates whether the 4-block Nambu structure is used.
    """
    if not xyz:
        E = np.sqrt(2*Jzz*(E+np.repeat(lams, int(E.shape[1]/2))))
    else:
        E = np.sqrt(2*Jzz*(E+np.repeat(np.repeat(lams, int(E.shape[1]/4)),2)))
    Ep = contract('ijk, ijk, ik->ij', V, np.conj(V), Jzz/E)
    return Ep


def rho_true_fast(weights, E, lams, Jzz, xyz):
    """Fast BZ-averaged spinon density (assumes equal A/B sublattice density)."""
    return integrate_fixed(I3_integrand, weights, E, lams, Jzz)*np.ones(2)

def rho_true(weights, E, V, lams, Jzz, xyz):
    """BZ-averaged spinon density resolved onto A and B sublattices.

    Returns a length-2 array [rho_A, rho_B] obtained by projecting the
    constraint integrand onto the sublattice-resolved eigenvector components.
    When xyz=True (Jpmpm != 0), the Nambu doubling is accounted for.
    """
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
    """Full site-resolved spinon density (not sublattice-averaged)."""
    return integrate_fixed(I3_integrand_site, weights, E, V, lams, Jzz, xyz)

#endregion

#region gradient find minLam
# Finding the minimum of the lowest spinon band over the BZ is needed to
# determine the minimum allowed Lagrange multiplier (below which the spectrum
# would become imaginary, i.e. the ansatz is unstable).

def Emin(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
         unitcell):
    """Lowest eigenvalue at a single k-point (scalar objective for scipy minimisation)."""
    k = k.reshape((1,3))
    return E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
                unitcell)[0][0,0]


#region find minlam scipy
def findminLam_scipy(M, K, tol, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell, BZres, kappa):
    """Find the minimum of the lowest spinon band over the Brillouin zone.

    First identifies candidate k-points on a coarse grid where the lowest
    eigenvalue is minimal, then refines each candidate with Nelder-Mead
    optimisation.  Returns the minimum eigenvalue (negated, so it becomes
    the minimum allowed lambda) and the corresponding k-point(s) qmin.

    These qmin momenta are the condensation wavevectors: if the spinon gap
    closes at qmin, Bose-Einstein condensation occurs there, signalling
    magnetic order.
    """
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
    """Solve for the Lagrange multiplier lambda via bisection.

    Adjusts lambda (one per sublattice) until the BZ-averaged spinon
    density rho equals the target occupation kappa.  The bisection
    operates between lamMin (= minimum allowed lambda to keep the
    spectrum real) and a large lamMax.

    Parameters
    ----------
    kappa : float
        Target constraint value (spinon occupation per site, typically S+1/2).
    tol : float
        Convergence tolerance on |rho - kappa|.
    lamM : ndarray, shape (2,)
        Lower bound for lambda on A and B sublattices.
    Jzz : float
        Ising coupling (sets the energy scale).
    weights : ndarray
        Quadrature weights for BZ integration.
    E : ndarray
        Eigenvalues of the mean-field Hamiltonian (without lambda).
    M : ndarray
        Full Hamiltonian matrices (used when inversion=False to re-diagonalise).
    xyz : bool
        Whether the Nambu-doubled (Jpmpm != 0) structure is used.
    inversion : bool
        If True, use the fast integrand (no re-diagonalisation); otherwise
        re-diagonalise at each lambda step.

    Returns
    -------
    lams : ndarray, shape (2,)
        Converged Lagrange multiplier [lambda_A, lambda_B].
    diverge : bool
        True if bisection failed to converge.
    """
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
        except Exception:
            lamMin = lams
        count = count + 1
        if (abs(lamMin - lamMax) < 5e-15).all() or count > 1e2:
            diverge=True
            break
    warnings.resetwarnings()
    return lams, diverge

#endregion

#region Mean field calculation
# Self-consistent calculation of the mean-field order parameters:
#   xi_{rs}  = <a^dag_r  b_s>   (hopping / normal Green's function)
#   chi_{rs} = <a_r  a_s>       (pairing / anomalous Green's function)
# These are computed from the spinon Green's function G(k) and then
# symmetrised according to the Projective Symmetry Group (PSG) constraints
# of the chosen gauge ansatz.

def chi_integrand(k, E, V, Jzz, unitcell):
    """BZ integrand for the pairing order parameter chi.

    Computes chi from the anomalous (off-diagonal) blocks of the spinon
    Green's function.  Returns two components [A, B] corresponding to
    the A-sublattice and B-sublattice anomalous correlators.
    """
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jlk->ijl', k, NNminus)
    ffactB = np.exp(-1j * ffact)
    size = int(E.shape[1]/4)
    B = contract('iab, ijl,jka, lkb->ikjl', green[:, 3*size:4*size, size:2*size], ffactB, unitcell, unitcell)/size
    ffactA = np.exp(1j * ffact)
    A = contract('iab, ijl,jka, lkb->ikjl', green[:, 2*size:3*size, 0:size], ffactA, unitcell, unitcell)/size
    return A, B

def chiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, chi_field, *args):
    """Compute the pairing mean-field parameter chi by BZ integration.

    Integrates chi_integrand over the BZ and then applies the PSG symmetry
    constraints via chi_field to produce the full chi tensor.
    """
    k = contract('ij,jk->ik', pts, BasisBZA)
    A, B = chi_integrand(k, E, V, Jzz, unitcellGraph)
    A = contract('ikjl,i->kjl', A, weights)
    B = contract('ikjl,i->kjl', B, weights)
    M1 = chi_field(n, n1, n2, unitcellCoord, B, A, *args)
    return M1

def xi_integrand(k, E, V, Jzz, unitcellGraph):
    """BZ integrand for the hopping order parameter xi.

    Computes xi from the normal (diagonal A-B) block of the spinon
    Green's function projected onto the sublattice basis.
    """
    green = green_pi(E, V, Jzz)
    ffact = contract('ik,jk->ij', k, NN)
    ffactA = np.exp(1j * ffact)
    size = int(E.shape[1]/4)
    A = contract('ika, ij,jka->ikj', green[:, 0:size, size:2*size], ffactA, unitcellGraph)/size
    return A
def xiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, xi_field, *args):
    """Compute the hopping mean-field parameter xi by BZ integration.

    Integrates xi_integrand over the BZ and applies PSG symmetry
    constraints via xi_field.
    """
    k = contract('ij,jk->ik', pts, BasisBZA)
    M = contract('ikj, i->kj', xi_integrand(k,E,V,Jzz,unitcellGraph), weights)
    M1 = xi_field(n, n1, n2, unitcellCoord, M, *args)
    return M1

def calmeanfield(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, xi_field, chi_field, params):
    """Compute both mean-field order parameters (chi and xi) in one pass."""
    chi = chiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, chi_field, params)
    return chi, xiCal(E, V, Jzz, n, n1, n2, pts, weights, unitcellCoord, unitcellGraph, xi_field, params)

def xiCalCondensed(rhos, qmin, n, n1, n2, unitcellCoord, unitcellGraph, xi_field, *args):
    """Condensate contribution to the hopping order parameter xi.

    When the spinon gap closes (Bose-Einstein condensation), a macroscopic
    occupation rho develops at the condensation wavevector qmin.  This
    function computes the additional contribution to xi from the condensate
    mode, which must be added to the BZ-integrated quantum fluctuation part.
    """
    k = contract('ij,jk->ik', qmin, BasisBZA)
    ffact = contract('ik,jk->ij', k, NN)
    ffactA = np.exp(1j * ffact)
    size = int(len(rhos)/4)
    A = contract('k, a, ij,jka->kj', np.conj(rhos[0:size]), rhos[size:2*size], ffactA, unitcellGraph)/size
    return A

def chiCalCondensed(rhos, qmin, n, n1, n2, unitcellCoord, unitcellGraph, chi_field, *args):
    """Condensate contribution to the pairing order parameter chi.

    Analogous to xiCalCondensed: computes the additional chi contribution
    from the macroscopic spinon condensate at qmin.
    """
    k = contract('ij,jk->ik', qmin, BasisBZA)
    size = int(len(rhos)/4)
    ffact = contract('ik,jlk->ijl', k, NNminus)
    ffactB = np.exp(-1j * ffact)
    B = contract('a, b, ijl,jka, lkb->kjl', rhos[3*size:4*size], rhos[size:2*size], ffactB, unitcellGraph, unitcellGraph)/size
    ffactA = np.exp(1j * ffact)
    A = contract('a, b, ijl,jka, lkb->kjl', rhos[2*size:3*size], rhos[0:size], ffactA, unitcellGraph, unitcellGraph)/size
    M1 = np.zeros((2, B.shape[0], B.shape[1], B.shape[2]), dtype=np.complex128)
    M1[0] = A
    M1[1] = B
    return M1

# endregion

#region graphing BZ
# Plotting the spinon dispersion along a standard high-symmetry path in the
# FCC Brillouin zone:  Gamma -> X -> W -> K -> Gamma -> L -> U -> W' -> X' -> Gamma.
# The path segments and their cumulative plot-axis positions are pre-computed
# in misc_helper.py.

# High-symmetry path segments and their plot-axis positions
BZ_PATH_SEGMENTS = [GammaX, XW, WK, KGamma, GammaL, LU, UW1, W1X1, X1Gamma]
BZ_PATH_POSITIONS = [
    (gGamma1, gX), (gX, gW), (gW, gK), (gK, gGamma2), (gGamma2, gL),
    (gL, gU), (gU, gW1), (gW1, gX1), (gX1, gGamma3)
]
BZ_TICK_POSITIONS = [gGamma1, gX, gW, gK, gGamma2, gL, gU, gW1, gX1, gGamma3]
BZ_TICK_LABELS = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W^\prime$', r'$X^\prime$', r'$\Gamma$']


def _compute_along_path(func, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi,
                        A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell,
                        extra_args=None):
    """Evaluate a function along each segment of the BZ high-symmetry path.

    Iterates over the 9 path segments (GammaX, XW, ..., X1Gamma) and
    collects results into a list.  This is the DRY helper that replaces
    the formerly duplicated per-segment evaluation in calDispersion,
    loweredge, upperedge, etc.
    """
    results = []
    for seg in BZ_PATH_SEGMENTS:
        if extra_args is not None:
            results.append(func(lams, seg, Jzz, Jpm, Jpmpm, h, n, *extra_args, theta, chi, xi,
                                A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell))
        else:
            results.append(func(lams, seg, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi,
                                A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell))
    return results


def _format_bz_axes(ax, color='b', adjust_W1=False):
    """Add high-symmetry point markers and labels to a BZ path plot axis."""
    for pos in BZ_TICK_POSITIONS:
        ax.axvline(x=pos, color=color, linestyle='dashed')
    xlabpos = list(BZ_TICK_POSITIONS)
    if adjust_W1:
        xlabpos[7] = xlabpos[7] * 1.02
    ax.set_xticks(xlabpos, BZ_TICK_LABELS)
    ax.set_xlim([0, gGamma3])


def _plot_segments(ax, data_list, color, zorder=None):
    """Plot data along BZ path segments."""
    for (start, end), data in zip(BZ_PATH_POSITIONS, data_list):
        kwargs = {'zorder': zorder} if zorder is not None else {}
        ax.plot(np.linspace(start, end, len(data)), data, color, **kwargs)


def dispersion_pi(lams, k, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, g, unitcell):
    """Physical spinon dispersion omega(k) = sqrt(2 * Jzz * E(k)).

    Returns the positive-definite spinon energies at the given k-points.
    These are the excitation energies above the mean-field ground state.
    """
    return np.sqrt(2 * Jzz * E_pi(k, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                  A_pi_rs_traced_pp_here, g, unitcell, True)[0])


def calDispersion(lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here,
                  axes, g, unitcell=piunitcell):
    """Plot all spinon dispersion bands along the BZ high-symmetry path.

    Each band (column of the eigenvalue array) is plotted as a blue line
    across the 9 path segments, with vertical dashed lines marking the
    high-symmetry points.
    """
    segments = _compute_along_path(dispersion_pi, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi,
                                   A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell)

    for i in range(segments[0].shape[1]):
        for (start, end), data in zip(BZ_PATH_POSITIONS, segments):
            axes.plot(np.linspace(start, end, len(data)), data[:, i], 'b')

    _format_bz_axes(axes, color='b')
    cluster = np.concatenate(segments)
    axes.set_ylim([np.min(cluster) * 0.5, np.max(cluster) * 1.2])
#endregion

#region lower and upper edges
# Two-spinon continuum boundaries.
# The dynamical spin structure factor S(q,omega) has spectral weight in the
# two-spinon continuum, bounded from below by the lower edge  omega_low(q)
# = min_k [omega(k) + omega(q-k)]  and from above by the upper edge
# omega_high(q) = max_k [omega(k) + omega(q-k)].

def minCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
           unitcell):
    """Lower edge of the two-spinon continuum at transfer momenta q.

    For each q, finds  min_K [ omega_0(K) + omega_0(K-q) ]  where omega_0
    is the lowest spinon band.
    """
    temp = np.zeros(len(q))
    mins = dispersion_pi(lams, K, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                         A_pi_rs_traced_pp_here, g, unitcell)[:, 0]
    for i in range(len(q)):
        temp[i] = np.min(
            dispersion_pi(lams, K - q[i], Jzz, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                          A_pi_rs_traced_pp_here, g, unitcell)[:, 0]
            + mins)
    return temp


def maxCal(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
           unitcell):
    """Upper edge of the two-spinon continuum at transfer momenta q.

    For each q, finds  max_K [ omega_top(K) + omega_top(K-q) ]  where
    omega_top is the highest spinon band.
    """
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
    """Compute both the lower and upper two-spinon continuum edges at once."""
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
    """Absolute lower bound of the two-spinon continuum for a given q."""
    Eq = np.sqrt(2 * Jzz * E_pi(K, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here, g)[0])
    Ek = np.sqrt(2 * Jzz * E_pi(K - q, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here, g)[0])
    return min(Eq[:,0]+Ek[:,0])

def DSSF_E_High(lams, q, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                A_pi_rs_traced_pp_here, g):
    """Absolute upper bound of the two-spinon continuum for a given q."""
    Eq = np.sqrt(2 * Jzz * E_pi(K, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here, g)[0])
    Ek = np.sqrt(2 * Jzz * E_pi(K - q, lams, Jpm, Jpmpm, h, n, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                                A_pi_rs_traced_pp_here, g)[0])
    return max(Eq[:,-1]+Ek[:,-1])

def DSSF_E_DOMAIN(lams, qmin, qmax, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                  A_pi_rs_traced_pp_here, g):
    """Energy window [E_low, E_high] of the two-spinon continuum."""
    return DSSF_E_Low(lams, qmin, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                      A_pi_rs_traced_pp_here, g) \
        , DSSF_E_High(lams, qmax, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                      A_pi_rs_traced_pp_here, g)


def loweredge(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
              unitcell, ax, color='w'):
    """Plot and return the lower edge of the two-spinon continuum along the BZ path."""
    segments = _compute_along_path(minCal, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi,
                                   A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell,
                                   extra_args=(K,))
    _plot_segments(ax, segments, color, zorder=8)
    _format_bz_axes(ax, color='w', adjust_W1=True)
    return np.concatenate(segments)

def upperedge(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g,
              unitcell, ax, color='w'):
    """Plot and return the upper edge of the two-spinon continuum along the BZ path."""
    segments = _compute_along_path(maxCal, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi,
                                   A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell,
                                   extra_args=(K,))
    _plot_segments(ax, segments, color, zorder=8)
    _format_bz_axes(ax, color='w', adjust_W1=True)
    return np.concatenate(segments)
def loweredge_data(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                   A_pi_rs_traced_pp_here, g, unitcell):
    """Return the lower-edge data without plotting."""
    segments = _compute_along_path(minCal, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi,
                                   A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell,
                                   extra_args=(K,))
    return np.concatenate(segments)

def upperedge_data(lams, Jzz, Jpm, Jpmpm, h, n, K, theta, chi, xi, A_pi_here, A_pi_rs_traced_here,
                   A_pi_rs_traced_pp_here, g, unitcell):
    """Return the upper-edge data without plotting."""
    segments = _compute_along_path(maxCal, lams, Jzz, Jpm, Jpmpm, h, n, theta, chi, xi,
                                   A_pi_here, A_pi_rs_traced_here, A_pi_rs_traced_pp_here, g, unitcell,
                                   extra_args=(K,))
    return np.concatenate(segments)


#endregion

#region greens function and energetics
# The spinon Green's function G(k, omega) encodes all single-particle
# correlations.  At omega=0 (equal-time) it gives the mean-field order
# parameters; at finite omega it enters the dynamical structure factor.

def green_pi(E, V, Jzz, omega=0):
    """Equal-time (omega=0) spinon Green's function G_{ab}(k).

    Constructed from the spectral decomposition:
        G_{ab}(k) = sum_n  V_{a,n}(k) V*_{b,n}(k)  *  Jzz / (omega^2 + E_n(k))
    At omega=0 this reduces to the zero-temperature correlator.
    """
    green = contract('ilk, ijk, ik->ijl', V, np.conj(V), Jzz / (omega**2+E))
    return green

def green_pi_branch(E, V, Jzz):
    """Band-resolved Green's function G_{n}(k) for each spinon branch.

    Unlike green_pi which sums over bands, this keeps the band index n
    separate, useful for computing band-resolved contributions to the
    dynamical structure factor.
    """
    green = contract('ilk, ijk, ik->ikjl', V, np.conj(V), Jzz / E)
    return green

#endregion

#region miscellaneous

def gap(M, lams):
    """Spinon gap: the minimum eigenvalue of M + lambda*I over all k."""
    temp = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
    E, _ = np.linalg.eigh(temp)
    return np.amin(E)


def EMAX(M, lams):
    """Maximum eigenvalue of M + lambda*I over all k (bandwidth estimate)."""
    temp = M + np.diag(np.repeat(lams, int(M.shape[1]/2)))
    E, _ = np.linalg.eigh(temp)
    return np.amax(E)

def graphing_M_setup(flux, n):
    """Set up the minimal unit cell and gauge connection for a given flux sector.

    Different Z2 flux configurations through the hexagonal plaquettes require
    different magnetic unit cells:
      - 0-flux:   1-site unit cell (no gauge phases)
      - pi-flux:  4-site unit cell with the standard A_pi gauge connection
      - pzzp/zppz: 2-site unit cells with partial pi-fluxes

    Returns the sublattice embedding matrices (unitCellgraph), the gauge
    connection array (A_pi_here), and the unit-cell coordinates.
    """
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
    """Set up the full 4-site unit cell for any flux sector.

    Unlike graphing_M_setup (which uses the minimal cell), this always
    uses the 4-site pi-flux unit cell and only changes the gauge
    connection A_pi_here.  Used for the self-consistent solver to
    maintain a uniform matrix structure across different flux sectors.
    """
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
    """Return the raw (unsymmetrised) hopping order parameter xi.

    No PSG constraints are applied -- all xi matrix elements are
    independent.  Used for debugging or when the symmetry is unknown.
    """
    #in the case of 110, three xi mf: xi0, xi1, xi3
    return xi0
def chi_unconstrained(n, n1, n2, unitcellCoord, chi0, chi0A, *args):
    """Return the raw (unsymmetrised) pairing order parameter chi."""
    return np.array([chi0, chi0A])
def xi_wo_field(n, n1, n2, unitcellcoord, xi, args):
    """PSG-symmetrised hopping order parameter xi in zero external field.

    All xi values are related by the space-group symmetry: on a given bond
    they differ only by the PSG phase factors exp(i pi n1 ...) associated
    with the gauge-enlarged translations.
    """
    #in the case of 110, three xi mf: xi0, xi1, xi3
    xi00 = np.real(np.mean(np.abs(xi)))
    mult = np.zeros((len(unitcellcoord),4),dtype=np.complex128)
    try:
        nS, = args
    except Exception:
        nS = 0
    for i in range(len(unitcellcoord)):
        mult[i] = np.array([xi00, xi00*np.exp(1j*np.pi*(nS+n1*(unitcellcoord[i,1]+unitcellcoord[i,2]))), xi00*np.exp(1j*np.pi*(nS+n1*unitcellcoord[i,2])), xi00*np.exp(1j*np.pi*nS)])
    return mult
def chi_wo_field(n, n1, n2, unitcellCoord, chi, chiA, *args):
    """PSG-symmetrised pairing order parameter chi in zero external field.

    The full 4x4 chi matrix on each unit cell is reconstructed from two
    independent amplitudes (chi000, chi001) using the PSG phase factors
    that depend on the unit-cell coordinates (r2, r3) and gauge quantum
    numbers n1.
    """
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
    """PSG-symmetrised xi in an external field with octupolar channel.

    The field direction n (h110 or h111) determines the residual symmetry
    group, which constrains the number of independent xi amplitudes:
      - [110] field: 3 independent xi values (xi0, xi1, xi2)
      - [111] field: 2 independent xi values (xi0, xi1)
    """
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
    """PSG-symmetrised chi in an external field with octupolar channel.

    The residual symmetry under the field determines how the A and B
    sublattice chi matrices are related.  For [110], the inversion psiI
    and translation phases psiIT1/psiIT2 are extracted from the ratio
    of the A and B correlators.  For [111], a C6 rotation phase psiC6
    relates all matrix elements.
    """
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
            except Exception:
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
            except Exception:
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
                psiS = chi0[2,3]/chi0A[2,3]
            except Exception:
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
    """PSG-symmetrised xi in an external field with dipolar channel.

    Same structure as xi_w_field_Octu but for the case where the dominant
    exchange is Jxx (dipolar rather than octupolar).
    """
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
    """PSG-symmetrised chi in an external field with dipolar channel.

    For the dipolar channel the PSG phase structure differs from the
    octupolar case.  The [110] field introduces additional sign factors
    (nI) that track the parity of the gauge transformation under inversion.
    """
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
            except Exception:
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
                psiC6 = chi0[0,0]/chi0A[0,0]
            except Exception:
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
            except Exception:
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
    """Self-consistent gauge mean-field theory solver for the pyrochlore lattice.

    Encapsulates all the state needed to solve the GMFT self-consistency
    equations: the exchange couplings (Jxx, Jyy, Jzz), the gauge flux
    configuration, external field, mean-field order parameters (xi, chi),
    Lagrange multipliers (lams), and the BZ integration grid.

    Typical usage::

        solver = piFluxSolver(Jxx, Jyy, Jzz, h=0.5, n=h111, flux=np.pi*np.ones(4))
        solver.solvemeanfield()  # run the self-consistency loop
        solver.graph(ax)         # plot the spinon dispersion
        gap = solver.gap()       # extract the spinon gap

    Parameters
    ----------
    Jxx, Jyy, Jzz : float
        Exchange coupling constants.  The code identifies the dominant
        (largest) coupling and rotates the Hamiltonian accordingly:
        Jpm = -(J_second + J_third)/4,  Jpmpm = (J_second - J_third)/4.
    *args
        Additional Projective Symmetry Group (PSG) parameters passed to
        the xi_field / chi_field symmetrisation functions.
    theta : float
        Angle parametrising in-plane field rotation.
    h : float
        External magnetic field strength (in meV or dimensionless units).
    n : ndarray, shape (3,)
        Field direction unit vector (default [1,1,0]).
    kappa : float
        Target spinon occupation per site (= 2S for spin-S).
    lam : float
        Initial Lagrange multiplier.
    BZres : int
        Number of grid points per direction for BZ integration.
    graphres : int
        Resolution for plotting along the high-symmetry path.
    tol : float
        Convergence tolerance for the constraint and mean-field iteration.
    flux : ndarray, shape (4,)
        Z2 gauge flux through the four distinct hexagonal plaquettes.
    intmethod : callable
        BZ integration quadrature (default: Gauss quadrature).
    gzz : float
        Landé g-factor for converting real magnetic field to coupling (T -> meV).
    Breal : bool
        If True, interpret h as a real magnetic field in Tesla.
    unconstrained : bool
        If True, skip PSG symmetrisation of the order parameters.
    g : float
        Fictitious Z2 gauge coupling.
    simplified : bool
        If True, use the minimal unit cell for the given flux sector.
    FF : bool
        If True, use the fractional-flux (non-pi) phase setup.
    """

    def __init__(self, Jxx, Jyy, Jzz, *args, theta=0, h=0, n=h110, kappa=2, lam=2, BZres=20, graphres=20, tol=1e-10, flux=np.zeros(4),
                 intmethod=gauss_quadrature_3D_pts, gzz=2.24, Breal=False, unconstrained=False, g=0, simplified=False, FF=False):
        self.intmethod = intmethod
        # Identify the dominant coupling axis and rotate the Hamiltonian
        # so that Jzz is always the largest coupling.
        J = np.array([Jxx, Jyy, Jzz])
        a = np.argmax(J)
        xx = np.mod(a+1,3)
        yy = np.mod(a+2,3)
        self.dominant = a
        self.Jzz = J[a]
        self.Jpm = -(J[xx] + J[yy]) / 4        # transverse exchange
        self.Jpmpm = (J[xx] - J[yy]) / 4       # XY anisotropy
        self.theta = theta
        self.kappa = kappa                      # constraint: <n_i> = kappa
        self.g = g                              # fictitious Z2 gauge coupling
        self.tol = tol
        self.lams = np.array([lam, lam], dtype=np.double)  # Lagrange multipliers [lambda_A, lambda_B]
        self.PSGparams = args
        # Select the symmetry-constrained form of xi and chi based on
        # whether a field is applied and which exchange axis dominates.
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

        # Convert real magnetic field (Tesla) to internal coupling if requested
        if Breal:
            self.h = 5.7883818060*10**(-2)*h*gzz
        else:
            self.h = h
        if a == 0:
            self.h = -1j*self.h   # imaginary field for dipolar channel
        self.inversion = True     # use fast constraint evaluation by default

        # Set up BZ integration grid (Gauss quadrature over [0,1]^3)
        self.pts, self.weights = self.intmethod(0, 1, 0, 1, 0, 1, BZres, BZres, BZres)

        self.minLams = np.zeros(2, dtype=np.double)
        self.BZres = BZres
        self.graphres = graphres

        self.toignore = np.array([], dtype=int)
        self.q = np.nan
        self.qmin = np.empty((1, 3))     # condensation wavevector(s)
        self.qmin[:] = 0
        self.qminWeight = np.zeros((1,))  # quadrature weight for condensate
        self.qminB = np.copy(self.qmin)   # qmin in Cartesian coords
        self.condensed = False            # True if spinon BEC has occurred
        if not FF:
            # Standard flux ansatz: determine the PSG equivalence class
            # and set up the gauge connection for the chosen flux configuration.
            self.n = n
            self.flux = flux
            self.A_pi_here, self.n1, self.n2 = determineEquivalence(n, flux)


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
            # Fractional-flux ansatz (non-pi): the unit cell and gauge
            # connection are determined by FFphase_setup for general flux angles.
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
        """Solve for the Lagrange multiplier lambda.

        If a=True, use the previously computed minLams as the lower bound;
        otherwise determine the lower bound from the minimum eigenvalue.
        """
        if a:
            return findlambda_pi(self.kappa, self.tol,self.minLams, self.Jzz, self.weights, self.E, self.MF, (not self.Jpmpm==0), self.inversion)
        else:
            A = -np.min(self.E)*np.ones(2)
            lams, d = findlambda_pi(self.kappa, self.tol, A+1e-16, self.Jzz, self.weights, self.E, self.MF, (not self.Jpmpm==0), self.inversion)
            return lams, d

    def findLambda_unconstrained(self):
        return findlambda_pi(self.kappa,self.tol, np.zeros(2), self.Jzz, self.weights, self.E, self.MF)


    def findminLam(self):
        """Find the minimum of the lowest spinon band over a dense BZ grid.

        Uses a coarse search on a 34-point grid followed by Nelder-Mead
        refinement.  Updates self.qmin (condensation wavevector) and
        self.minLams (minimum allowed Lagrange multiplier).
        """
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
        """Compute the hopping order parameter xi from the current state.

        Combines the BZ-integrated contribution (from the Green's function)
        with the condensate contribution (if the system is condensed).
        """
        E = np.sqrt(2*self.Jzz*(self.E+np.repeat(np.repeat(self.lams,int(self.E.shape[1]/4)),2)))
        xi = xiCal(E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.PSGparams)
        xiC = xiCalCondensed(self.rhos, self.qmin, self.n, self.n1, self.n2, self.unitcellCoord, self.unitCellgraph, self.xi_field, self.PSGparams)
        return xi + xiC
    
    def solvechifield(self):
        """Compute the pairing order parameter chi from the current state.

        Combines the BZ-integrated contribution with the condensate
        contribution, analogous to solvexifield.
        """
        E = np.sqrt(2*self.Jzz*(self.E+np.repeat(np.repeat(self.lams,int(self.E.shape[1]/4)),2)))
        chi = chiCal(E, self.V, self.Jzz, self.n, self.n1, self.n2, self.pts, self.weights, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.PSGparams)
        chiC = chiCalCondensed(self.rhos, self.qmin, self.n, self.n1, self.n2, self.unitcellCoord, self.unitCellgraph, self.chi_field, self.PSGparams)
        return chi + chiC
    
    def updateMF(self):
        """Rebuild the Hamiltonian matrix from the current order parameters.

        Reconstructs M(k) with the latest xi/chi and re-diagonalises.
        If diagonalisation fails (e.g. near a singularity), falls back
        to resetting lambda and recomputing the order parameters.
        """
        self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                      self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
        try:
            self.E, self.V = np.linalg.eigh(self.MF)
        except Exception:
            self.lams = (-np.min(self.E)+1e-14)*np.ones(2)
            self.xi = self.solvexifield()
            self.chi = self.solvechifield()
            self.MF = M_pi(self.pts, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                          self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
            self.E, self.V = np.linalg.eigh(self.MF)
    def xiSubroutine(self, tol, GS, pcon=False):
        """Inner convergence loop for the hopping order parameter xi.

        Repeatedly updates xi, rebuilds the Hamiltonian, and re-solves
        for lambda until xi converges (or the iteration limit is reached).
        """
        limit = 5
        print("Begin Xi Subroutine")
        count = 0
        pb = False
        while True:
            xilast = np.copy(self.xi)
            self.xi = self.solvexifield()
            self.updateMF()
            GS, diverge = self.solvemufield()
            if np.abs(GS) > 1e1 or diverge:
                pb = True
            count = count + 1
            if ((abs(self.xi - xilast) < tol).all()) or count > limit:
                break
        print("Xi Subrountine ends. Exiting Energy is: "+ str(GS) + " Took " + str(count) + " cycles.")
        return GS, pb
    def chiSubroutine(self, tol, GS, pcon=False):
        """Inner convergence loop for the pairing order parameter chi.

        Analogous to xiSubroutine but for the anomalous (pairing) channel.
        """
        limit = 5
        print("Begin Chi Subroutine")
        count = 0
        pb = False
        while True:
            chilast = np.copy(self.chi)
            self.chi = self.solvechifield()
            self.updateMF()
            GS, diverge = self.solvemufield()
            if np.abs(GS) > 1e1 or diverge:
                pb = True
            count += 1
            if ((abs(self.chi - chilast) < tol).all()) or count > limit:
                break
        print("Chi Subrountine ends. Exiting Energy is: "+ str(GS) + " Took " + str(count) + " cycles.")
        return GS, pb

    def solvemufield(self, a=True):
        """Solve for lambda and check for condensation.

        This is one "inner iteration" of the self-consistency loop:
        find the minimum eigenvalue (if a=True), solve for lambda,
        check/set the condensation flag, and return the mean-field energy.
        """
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

    def solvemeanfield(self, solve_all=True, Fast=False, ref_energy=0):
        """Run the full self-consistency loop to convergence.

        Parameters
        ----------
        solve_all : bool
            If True, update xi and chi simultaneously each iteration
            (solvemeanfield_all).  If False, alternate xi and chi updates
            in separate subroutines (solvemeanfield_seq).
        Fast : bool
            If True, skip the minLam search each iteration (faster but
            less robust; uses solvemeanfield_fast).
        ref_energy : float
            Reference energy for the fast solver to detect condensation.
        """
        if Fast:
            self.solvemeanfield_fast(ref_energy)
        else:
            if solve_all:
                return self.solvemeanfield_all()
            else:
                return self.solvemeanfield_seq()
    def solvemeanfield_seq(self, tol=1e-13):
        """Sequential self-consistency: alternate xi and chi subroutines.

        In each outer iteration, first converge xi (with chi fixed) then
        converge chi (with xi fixed).  This is more stable but slower.
        For Jpmpm=0, xi and chi are zero by symmetry and only lambda needs
        to be solved.
        """
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
                chilast, xilast = np.copy(self.chi), np.copy(self.xi)
                GS, pcon = self.xiSubroutine(tol, GS, pcon)
                GS, pcon = self.chiSubroutine(tol, GS, pcon)
                print("Iteration #"+str(count))
                count += 1
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
        """Simultaneous self-consistency: update xi and chi together each iteration.

        This converges faster than the sequential variant but may be less
        stable near phase boundaries.
        """
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
            GS, d = self.solvemufield()

            print("Initialization Routine Ends. Starting Parameters: GS="+ str(GS) + " xi0= " + str(self.xi[0]) + " chi0= " + str(self.chi[0,0]))
            count = 0
            while True:
                chilast, xilast = np.copy(self.chi), np.copy(self.xi)
                self.xi = self.solvexifield()
                self.chi = self.solvechifield()
                self.updateMF()
                GS, diverge = self.solvemufield()
                print("Iteration #"+str(count), GS, self.condensed)
                count += 1
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
        """Fast self-consistency loop without re-searching for minLam each step.

        Uses a=False in solvemufield to skip the expensive BZ minimisation.
        The condensation check is done at the end by comparing to ref_energy.
        """
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
                chilast, xilast = np.copy(self.chi), np.copy(self.xi)
                self.xi = self.solvexifield()
                self.chi = self.solvechifield()
                self.updateMF()
                GS, diverge = self.solvemufield(False)
                print("Iteration #" + str(count), GS, self.condensed)
                count += 1
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
        if self.condensed:
            self.toignore = indextoignore_tol(self.pts, self.qmin, 1e-10)

    def low(self):
        E, V = np.linalg.eigh(self.MF)
        cond = np.argmin(E[:, 0])
        return self.bigB[cond], E[cond][0]

    def set_condensed(self):
        """Check whether the spinon gap has closed (Bose-Einstein condensation).

        If the gap (lambda - minLam) is smaller than a threshold proportional
        to (deltamin / BZres^3)^2, the system is flagged as condensed, meaning
        the spinon mode at qmin has macroscopic occupation and the ground
        state has magnetic order.
        """
        A = -self.minLams[0] + self.lams[0]
        if A < (deltamin / (self.BZres**3)) ** 2:
            self.condensed = True
        else:
            self.condensed = False

    def set_delta(self):
        """Compute the condensate wavefunction (spinon BEC order parameter).

        The condensate amplitude rho is determined from the deficit
        kappa - rho_site (the 'missing' constraint density that must be
        supplied by the macroscopic mode).  The condensate wavefunction
        is proportional to the null-space eigenvector of M(qmin) + lambda*I.
        """
        rho = np.sqrt(self.kappa - self.rho_site())
        from scipy.linalg import null_space
        M_kc = M_pi(self.qmin, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                       self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)
        M_kc = M_kc + np.diag(np.repeat(self.lams, int(M_kc.shape[1]/2)))
        E, V = np.linalg.eigh(M_kc)
        self.rhos = np.linalg.norm(rho) * V[0,:,0]
        self.rhos = self.rhos
    def condensation_check(self):
        """Full condensation diagnostic: find minLam, solve lambda, set condensate."""
        self.findminLam()
        self.lams, d = self.findLambda(True)
        self.set_condensed()
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
        return np.sqrt(2 * self.Jzz *
                       E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi,
                            self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                            self.unitCellgraph)[0])


    def dispersion(self, k):
        """Evaluate the physical spinon dispersion omega(k) at arbitrary k-points."""
        return dispersion_pi(self.lams, k, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi,
                             self.xi, self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                             self.unitCellgraph)

    def LV_zero(self, k):
        """Return eigenvalues and eigenvectors of the Hamiltonian at k with current lambda."""
        return E_pi(k, self.lams, self.Jpm, self.Jpmpm, self.h, self.n, self.theta, self.chi, self.xi, self.A_pi_here,
                    self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g, self.unitCellgraph)

    def GS(self):
        """Mean-field ground-state energy per site.

        E_GS = <H_MF> = BZ-average of spinon energies minus kappa*lambda.
        """
        try:
            E = np.dot(self.E_pi_mean(self.pts), self.weights) - self.kappa*self.lams[0]
        except Exception:
            self.lams = (-np.min(self.E)+1e-16)*np.ones(2)
            E = np.dot(self.E_pi_mean(self.pts), self.weights) - self.kappa*self.lams[0]
        return E

    def MFE_condensed(self):
        return 0
    def MFE(self):
        """Total mean-field energy including any condensate contribution."""
        Ep = self.GS() + self.MFE_condensed()
        return np.real(Ep)

    def graph(self, axes):
        """Plot the full spinon band structure along the BZ high-symmetry path."""
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
        """Global energy window [min, max] of the two-spinon continuum.

        Searches a dense BZ grid to find the absolute minimum of the
        lower edge (= twice the single-spinon gap) and the absolute
        maximum of the upper edge (= twice the bandwidth).
        """
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
        """Compute (and optionally plot) the lower two-spinon continuum edge."""
        K = contract('ij,jk->ik', self.pts, BasisBZA)
        if show:
            result = loweredge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, K, self.theta, self.chi, self.xi,
                               self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                               self.unitCellgraph, ax, color)
            plt.show()
        else:
            result = loweredge_data(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, K, self.theta, self.chi, self.xi,
                                    self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                                    self.unitCellgraph)
        return result

    def graph_upperedge(self, show, ax=plt, color='w'):
        """Compute (and optionally plot) the upper two-spinon continuum edge."""
        K = contract('ij,jk->ik', self.pts, BasisBZA)
        if show:
            result = upperedge(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, K, self.theta, self.chi, self.xi,
                               self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                               self.unitCellgraph, ax, color)
            plt.show()
        else:
            result = upperedge_data(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, K, self.theta, self.chi, self.xi,
                                    self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                                    self.unitCellgraph)
        return result


    def loweredge(self):
        K = contract('ij,jk->ik', self.pts, BasisBZA)
        return loweredge_data(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, K, self.theta, self.chi, self.xi,
                              self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                              self.unitCellgraph)

    def upperedge(self):
        K = contract('ij,jk->ik', self.pts, BasisBZA)
        return upperedge_data(self.lams, self.Jzz, self.Jpm, self.Jpmpm, self.h, self.n, K, self.theta, self.chi, self.xi,
                              self.A_pi_here, self.A_pi_rs_traced_here, self.A_pi_rs_traced_pp_here, self.g,
                              self.unitCellgraph)

    def green_pi(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi(E, V, self.Jzz)

    def green_pi_branch(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi_branch(E, V, self.Jzz), E

    def green_pi_reduced(self, k, cartesian=False):
        """Green's function evaluated with the minimal (reduced) unit cell.

        Re-derives the gauge connection for the minimal cell and computes
        the Green's function there.  Useful for computing the SSSF where
        the reduced BZ is sufficient.
        """
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
        """Band-resolved Green's function in the reduced unit cell.

        Also returns the dispersion E and the gauge arrays needed for
        computing the dynamical structure factor with proper form factors.
        """
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
        """Magnetic order parameter: <a^dag b> averaged over the condensate mode.

        Non-zero only when the system is condensed (Jpmpm != 0 and gap closed).
        """
        if self.condensed and not self.Jpmpm == 0:
            size = int(len(self.rhos)/4)
            return np.mean(contract('i,i->i',np.conj(self.rhos[0:size]),self.rhos[size:2*size]))
        else:
            return 0
    def gap(self):
        """Single-spinon gap: energy of the lowest spinon mode above the ground state."""
        return np.sqrt(2*self.Jzz*(np.min(self.E)+self.lams[0]))

    def mag_integrand(self, k):
        """BZ integrand for the sublattice magnetisation <S^z> in the local frame."""
        E = np.sqrt(2 * self.Jzz * (self.E + self.lams[0]))
        green = green_pi(E, self.V, self.Jzz)

        ffact = contract('ik, jk->ij', k, NN)
        ffact = np.exp(1j * ffact)
        l = len(self.A_pi_here)
        magp = contract('ika, ij,jka, kj->ika', green[:, 0:l, l:2*l], ffact, self.unitCellgraph, np.exp(1j * self.A_pi_here)) / l
        return np.real(magp)
    def magnetization(self):
        """Compute the average sublattice magnetisation by BZ integration.

        Returns NaN if the system is condensed (the uniform magnetisation
        formula is not valid when there is a macroscopic condensate).
        """
        sz = np.einsum('kru,k->ru',self.mag_integrand(self.pts), self.weights)
        print(sz)
        mag = np.mean(sz)
        if self.condensed:
            mag = np.nan
        return np.real(mag)