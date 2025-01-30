# import pyrochlore_dispersion_pi_gang_chen as pygang
import numpy as np

from misc_helper import *
import matplotlib.pyplot as plt
from matplotlib import cm
import pyrochlore_gmft as pycon
from mpi4py import MPI
import os
import pathlib
import netCDF4 as nc


def quantum_fisher_information(temp, filename, tosave):
    D = np.loadtxt(filename)
    size = int(np.sqrt(D.shape[0]))
    omega = np.linspace(0, 1, D.shape[1])
    if not temp == 0:
        beta = 1/(temp)
        print(beta)
        factor1 = 4*np.tanh(omega*beta/2)
        factor2 = 1-np.exp(-beta*omega)
        toint = contract('w, w, kw->kw',factor1, factor2, D)/(2*np.pi)
        results = np.trapezoid(toint, omega,axis=1)

        results = results.reshape((size, size))
        np.savetxt(tosave, results)
    else:
        results = np.trapezoid(D, omega,axis=1)
        results = 4 * results.reshape((size,size))/(2*np.pi)
    plt.imshow(results, extent=[-2.5,2.5,-2.5,2.5], origin='lower')
    plt.colorbar()
    plt.show()


def quantum_fisher_information_K(D, temp):
    size = int(np.sqrt(D.shape[0]))
    omega = np.linspace(0, 10, D.shape[1])
    if not temp == 0:
        beta = 1/(temp)
        factor1 = 4*np.tanh(omega*beta/2)
        factor2 = 1-np.exp(-beta*omega)
        toint = contract('w, w, kw->kw',factor1, factor2, D)/(2*np.pi)
        results = np.trapezoid(toint, omega,axis=1)
    else:
        results = np.trapezoid(D, omega,axis=1)
        results = 4 * results.reshape((size,size))/(2*np.pi)
    return results


def deltas_beta(Ek, Eq, omega, beta, tol):
    size = Ek.shape[1]
    omsize = len(omega)
    Ekenlarged = contract('ik,j,w->iwkj', Ek, np.ones(size),np.ones(omsize))
    Eqenlarged = contract('ik,j,w->iwjk', Eq, np.ones(size),np.ones(omsize))
    omegaenlarged = contract('i, w, j, k->iwjk', np.ones(len(Ek)), omega, np.ones(size), np.ones(size))
    A = contract('ia, ib, iwab->iwab', 1 + bose(beta, Ek), 1 + bose(beta, Eq),
                 cauchy(omegaenlarged - Ekenlarged - Eqenlarged, tol))
    B = contract('ia, ib, iwab->iwab', 1 + bose(beta, Ek), bose(beta, Eq), cauchy(omegaenlarged - Ekenlarged + Eqenlarged, tol))
    C = contract('ia, ib, iwab->iwab', bose(beta, Ek), 1 + bose(beta, Eq), cauchy(omegaenlarged + Ekenlarged - Eqenlarged, tol))
    D = contract('ia, ib, iwab->iwab', bose(beta, Ek), bose(beta, Eq), cauchy(omegaenlarged + Ekenlarged + Eqenlarged, tol))
    return A + B + C + D

def SSSF_finite_temp_factor(Ek, Eq, beta):
    size = Ek.shape[1]
    A = contract('ia, ib->iab', 1 + bose(beta, Ek), 1 + bose(beta, Eq))
    B = contract('ia, ib->iab', 1 + bose(beta, Ek), bose(beta, Eq))
    C = contract('ia, ib->iab', bose(beta, Ek), 1 + bose(beta, Eq))
    D = contract('ia, ib->iab', bose(beta, Ek), bose(beta, Eq))
    return A + B + C + D

def deltas(Ek, Eq, omega, tol):
    size = Ek.shape[1]
    omsize = len(omega)
    Ekenlarged = contract('ik,j,w->iwkj', Ek, np.ones(size),np.ones(omsize))
    Eqenlarged = contract('ik,j,w->iwjk', Eq, np.ones(size),np.ones(omsize))
    omegaenlarged = contract('i, w, j, k->iwjk', np.ones(len(Ek)), omega, np.ones(size), np.ones(size))
    A = cauchy(omegaenlarged-Ekenlarged-Eqenlarged, tol)
    return A

def deltas_pairing(Ek, omega, tol):
    size = Ek.shape[1]
    omsize = len(omega)
    Ekenlarged = contract('ik,w->iwk', 2*Ek, np.ones(omsize))
    omegaenlarged = contract('i, w, k->iwk', np.ones(len(Ek)), omega, np.ones(size))
    A = cauchy(omegaenlarged-Ekenlarged, tol)
    return A

# region DSSF

def Spm_Spp_omega(Ks, Qs, q, omega, tol, pyp0, beta=0):
    greenpK, tempE, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here, unitcell = pyp0.green_pi_branch_reduced(Ks, True)
    greenpQ, tempQ, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here, unitcell = pyp0.green_pi_branch_reduced(Qs, True)

    if pyp0.Jpmpm ==0:
        size = int(tempE.shape[1]/2)
    else:
        size = int(tempE.shape[1]/4)

    # Kreal = contract('ij,jk->ik',Ks-q/2, BasisBZA)
    Kreal = Ks-q/2
    if beta == 0:
        deltapm = deltas(tempE, tempQ, omega, tol)
    else:
        deltapm = deltas_beta(tempE, tempQ, omega, 1/beta, tol)

    ffact = contract('ik, jlk->ijl', Kreal, NNminus)
    ffactpm = np.exp(1j * ffact)
    ffact = contract('ik, jlk->ijl', Kreal, NNplus)
    ffactpp = np.exp(1j * ffact)
    Spm = contract('ioab, ipyx, iwop, abjk, jax, kby, ijk->wijk', greenpK[:, :, 0:size, 0:size], greenpQ[:, :, size:2*size, size:2*size],
                   deltapm, A_pi_rs_rsp_here, unitcell, unitcell,
                   ffactpm) / size**2

    Spp = contract('ioay, ipbx, iwop, abjk, jax, kby, ijk->wijk', greenpK[:, :, 0:size, size:2*size], greenpQ[:, :, 0:size, size:2*size],
                   deltapm, A_pi_rs_rsp_pp_here, unitcell, unitcell,
                   ffactpp) / size**2
    if not pyp0.Jpmpm == 0:
        SppA = contract('ioab, ipyx, iwop, abjk, jax, kby, ijk->wijk', greenpK[:, :, 0:size, 2*size:3*size],
                       greenpQ[:, :, 3*size:4*size, size:2*size],
                       deltapm, A_pi_rs_rsp_pp_here, unitcell, unitcell,
                       ffactpm) / size**2
        Spp = Spp + SppA
    return Spm, Spp

def DSSF_core(q, omega, pyp0, tol):
    Ks = pyp0.pts
    Qs = Ks - q
    Spm, Spp = Spm_Spp_omega(Ks, Qs, q, omega, tol, pyp0, pyp0.lams)

    Szz = (np.real(Spm) + np.real(Spp)) / 2
    Sxx = (np.real(Spm) - np.real(Spp)) / 2

    Sglobalzz = contract('wijk,jk, i->w', Szz, g(q), pyp0.weights)
    Szz = contract('wijk, i->w', Szz, pyp0.weights)

    Sglobalxx = contract('wijk,jk, i->w', Sxx, g(q), pyp0.weights)
    Sxx = contract('wijk, i->w', Sxx, pyp0.weights)
    return Szz, Sglobalzz, Sxx, Sglobalxx
def graph_DSSF(pyp0, E, K, tol, rank, size):
    comm = MPI.COMM_WORLD
    n = len(K) / size

    left = int(rank * n)
    right = int((rank + 1) * n)

    currsize = right - left

    sendtemp = np.zeros((currsize, len(E)), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, len(E)), dtype=np.float64)
    sendtemp2 = np.zeros((currsize, len(E)), dtype=np.float64)
    sendtemp3 = np.zeros((currsize, len(E)), dtype=np.float64)

    currK = K[left:right]

    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None

    if rank == 0:
        rectemp = np.zeros((len(K), len(E)), dtype=np.float64)
        rectemp1 = np.zeros((len(K), len(E)), dtype=np.float64)
        rectemp2 = np.zeros((len(K), len(E)), dtype=np.float64)
        rectemp3 = np.zeros((len(K), len(E)), dtype=np.float64)

    for i in range(currsize):
        sendtemp[i], sendtemp1[i], sendtemp2[i], sendtemp3[i] = DSSF_core(currK[i], E, pyp0, tol)

    sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0] * sendtemp1.shape[1], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0] * sendtemp2.shape[1], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0] * sendtemp3.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)

    return rectemp, rectemp1, rectemp2, rectemp3

def TWOSPINON_core_dumb(E, Jpm, h, hn, flux, BZres, tol, Jpmpm=0):
    pyp0 = pycon.piFluxSolver(-2*Jpm-2*Jpmpm,1,-2*Jpm+2*Jpmpm,h=h,n=hn,flux=flux,BZres=BZres)
    pyp0.solvemeanfield()
    Ks = pyp0.pts
    Ek = pyp0.E_pi_reduced(Ks)
    size = Ek.shape[1]
    omsize = len(E)
    Ekenlarged = contract('ik,w->iwk', Ek, np.ones(omsize))
    omegaenlarged = contract('i, w, k->iwk', np.ones(len(Ek)), E, np.ones(size))
    A = contract('iwk->w',cauchy(omegaenlarged-Ekenlarged, tol))
    A = np.convolve(A, A)
    A = A/np.linalg.norm(A)
    return A
def graph_2S_rho_dumb(E, Jpm, h, hn, flux, BZres, tol, rank, size):
    comm = MPI.COMM_WORLD
    if isinstance(Jpm, np.ndarray):
        n = len(Jpm) / size

        left = int(rank * n)
        right = int((rank + 1) * n)

        currsize = right - left
        sendtemp = np.zeros((currsize, len(E)), dtype=np.float64)
        currK = Jpm[left:right]
        rectemp = None
        if rank == 0:
            rectemp = np.zeros((len(Jpm), len(E)), dtype=np.float64)
        for i in range(currsize):
            print(currK[i])
            sendtemp[i] = TWOSPINON_core_dumb(E, currK[i], h, hn, flux, BZres, tol)
        sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1], 0))
        comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
        return rectemp
    else:
        n = len(h) / size

        left = int(rank * n)
        right = int((rank + 1) * n)

        currsize = right - left
        sendtemp = np.zeros((currsize, len(E)), dtype=np.float64)
        currK = h[left:right]
        rectemp = None
        if rank == 0:
            rectemp = np.zeros((len(h), len(E)), dtype=np.float64)
        for i in range(currsize):
            print(currK[i])
            sendtemp[i] = TWOSPINON_core_dumb(E, Jpm, currK[i], hn, flux, BZres, tol)
        sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1], 0))
        comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
        return rectemp


def graph_2S_rho(E, Jpm, h, hn, BZres, rank, size, tol, Jpmpm=0):
    comm = MPI.COMM_WORLD
    n = len(h) / size

    if Jpm == -0.03:
        if (hn==h111).all():
            change = 0.33
        elif (hn==h001).all():
            change = 0.17
        else:
            change = 0.2
    else:
        change = 0

    left = int(rank * n)
    right = int((rank + 1) * n)

    currsize = right - left
    sendtemp = np.zeros((currsize, 2*len(E)-1), dtype=np.float64)
    currK = h[left:right]
    rectemp = None
    if rank == 0:
        rectemp = np.zeros((len(h), 2*len(E)-1), dtype=np.float64)
    for i in range(currsize):
        # if Jpm == -0.3:
        #     flux = np.ones(4) * np.pi
        # elif Jpm == 0.03:
        #     flux = np.zeros(4)
        # elif currK[i] > change:
        #     if (hn==h110).all():
        #         flux = np.array([0,0,np.pi,np.pi])
        #     else:
        #         flux = np.zeros(4)
        # else:
        flux = np.ones(4) * np.pi
        sendtemp[i] = TWOSPINON_core_dumb(E, Jpm, currK[i], hn, flux, BZres, tol, Jpmpm)
    sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1], 0))
    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    return rectemp


def graph_2S_rho_111_a(E, Jpm, h, hn, BZres, rank, size, tol):
    comm = MPI.COMM_WORLD
    n = len(Jpm) / size

    left = int(rank * n)
    right = int((rank + 1) * n)

    currsize = right - left
    sendtemp = np.zeros((currsize, len(E)), dtype=np.float64)
    currK = Jpm[left:right]
    rectemp = None
    if rank == 0:
        rectemp = np.zeros((len(Jpm), len(E)), dtype=np.float64)
    for i in range(currsize):
        if currK[i] > -0.02:
            flux = np.zeros(4)
        else:
            flux = np.ones(4) * np.pi
        sendtemp[i] = TWOSPINON_core_dumb(E, currK[i], h, hn, flux, BZres, tol)
    sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1], 0))
    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    return rectemp

def DSSF_core_pedantic(q, omega, pyp0, tol, beta=0):
    Ks = pyp0.pts
    Ks = contract('ia,ak->ik', Ks, BasisBZA)
    Qs = Ks - q
    Spm, Spp = Spm_Spp_omega(Ks, Qs, q, omega, tol, pyp0, beta)

    Szz = (np.real(Spm) + np.real(Spp)) / 2
    Sxx = (np.real(Spm) - np.real(Spp)) / 2

    Sglobalzz = contract('wijk,jk, i->wjk', Szz, g(q), pyp0.weights)
    Szz = contract('wijk, i->wjk', Szz, pyp0.weights)

    Sglobalxx = contract('wijk,jk, i->wjk', Sxx, gx(q), pyp0.weights)
    Sxx = contract('wijk, i->wjk', Sxx, pyp0.weights)
    return Szz, Sglobalzz, Sxx, Sglobalxx

def graph_DSSF_pedantic(pyp0, E, K, tol, rank, size, beta=0):
    comm = MPI.COMM_WORLD
    n = len(K) / size

    left = int(rank * n)
    right = int((rank + 1) * n)

    currsize = right - left

    sendtemp = np.zeros((currsize, len(E), 4 ,4), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, len(E), 4 ,4), dtype=np.float64)
    sendtemp2 = np.zeros((currsize, len(E), 4 ,4), dtype=np.float64)
    sendtemp3 = np.zeros((currsize, len(E), 4 ,4), dtype=np.float64)

    currK = K[left:right]

    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None

    if rank == 0:
        rectemp = np.zeros((len(K), len(E), 4 ,4), dtype=np.float64)
        rectemp1 = np.zeros((len(K), len(E), 4 ,4), dtype=np.float64)
        rectemp2 = np.zeros((len(K), len(E), 4 ,4), dtype=np.float64)
        rectemp3 = np.zeros((len(K), len(E), 4 ,4), dtype=np.float64)

    for i in range(currsize):
        sendtemp[i], sendtemp1[i], sendtemp2[i], sendtemp3[i] = DSSF_core_pedantic(currK[i], E, pyp0, tol, beta)
        # sendtemp[i] = DSSF_core_pedantic(currK[i], E, pyp0, tol)

    sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1]*16, 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0] * sendtemp1.shape[1]*16, 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0] * sendtemp2.shape[1]*16, 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0] * sendtemp3.shape[1]*16, 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)

    return rectemp, rectemp1, rectemp2, rectemp3
    # return rectemp


# endregion

# region SSSF

def SpmSpp(K, Q, q, pyp0, beta=0):
    if beta == 0:
        return SpmSpp_zerotemp(K,Q,q,pyp0)
    else:
        return SpmSpp_finite_temp(K,Q,q,pyp0, beta)

def SpmSpp_zerotemp(K, Q, q, pyp0):
    greenpK = pyp0.green_pi(K)
    greenpQ = pyp0.green_pi(Q)
    Kreal = contract('ij,jk->ik',K-q/2, BasisBZA)

    if pyp0.Jpmpm ==0:
        size = int(greenpK.shape[1]/2)
    else:
        size = int(greenpK.shape[1]/4)

    ffactpm = np.exp(1j * contract('ik, jlk->ijl', Kreal, NNminus))
    ffactpp = np.exp(1j * contract('ik, jlk->ijl', Kreal, NNplus))

    Spm = contract('iab, iyx, abjk, jax, kby, ijk->ijk', greenpK[:, 0:size, 0:size], greenpQ[:, size:2*size, size:2*size], pyp0.A_pi_rs_rsp_here,
                   piunitcell, piunitcell,
                   ffactpm)/size**2

    Spp = contract('iay, ibx, abjk, jax, kby, ijk->ijk', greenpK[:, 0:size, size:2*size], greenpQ[:, 0:size, size:2*size], pyp0.A_pi_rs_rsp_pp_here,
                   piunitcell, piunitcell,
                   ffactpp)/size**2
    if not pyp0.Jpmpm == 0:
        SppA = contract('iab, iyx, abjk, jax, kby, ijk->ijk', greenpK[:, 0:size, 2*size:3*size], greenpQ[:, 3*size:4*size, size:2*size], pyp0.A_pi_rs_rsp_pp_here,
                   piunitcell, piunitcell,
                   ffactpm)/size**2
        Spp = Spp + SppA
    return Spm, Spp

def SpmSpp_finite_temp(K, Q, q, pyp0, beta):
    greenpK, tempE, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here, unitcell = pyp0.green_pi_branch_reduced(K, True)
    greenpQ, tempQ, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here, unitcell = pyp0.green_pi_branch_reduced(Q, True)

    if pyp0.Jpmpm ==0:
        size = int(tempE.shape[1]/2)
    else:
        size = int(tempE.shape[1]/4)

    # Kreal = contract('ij,jk->ik',Ks-q/2, BasisBZA)
    Kreal = K-q/2
    deltapm = SSSF_finite_temp_factor(tempE, tempQ, beta)

    ffact = contract('ik, jlk->ijl', Kreal, NNminus)
    ffactpm = np.exp(1j * ffact)
    ffact = contract('ik, jlk->ijl', Kreal, NNplus)
    ffactpp = np.exp(1j * ffact)
    Spm = contract('ioab, ipyx, iwop, abjk, jax, kby, ijk->wijk', greenpK[:, :, 0:size, 0:size], greenpQ[:, :, size:2*size, size:2*size],
                   deltapm, A_pi_rs_rsp_here, unitcell, unitcell,
                   ffactpm) / size**2

    Spp = contract('ioay, ipbx, iwop, abjk, jax, kby, ijk->wijk', greenpK[:, :, 0:size, size:2*size], greenpQ[:, :, 0:size, size:2*size],
                   deltapm, A_pi_rs_rsp_pp_here, unitcell, unitcell,
                   ffactpp) / size**2
    if not pyp0.Jpmpm == 0:
        Spm = contract('ioab, ipyx, iwop, abjk, jax, kby, ijk->wijk', greenpK[:, :, 0:size, 2*size:3*size],
                       greenpQ[:, :, 3*size:4*size, size:2*size],
                       deltapm, A_pi_rs_rsp_pp_here, unitcell, unitcell,
                       ffactpm) / size**2
    return Spm, Spp


def SSSF_core_pedantic(q, v, pyp0):
    Ks = pyp0.pts
    Qs = Ks - q

    Spm, Spp = SpmSpp(Ks, Qs, q, pyp0)

    if pyp0.dominant == 0:
        Sxx = (np.real(Spm) + np.real(Spp)) / 2
        Szz = (np.real(Spm) - np.real(Spp)) / 2
    else:
        Szz = (np.real(Spm) + np.real(Spp)) / 2
        Sxx = (np.real(Spm) - np.real(Spp)) / 2

    qreal = contract('j,jk->k',q, BasisBZA)

    Szzglobal = contract('ijk, jk,i->jk', Szz, g(qreal), pyp0.weights)
    Sxxglobal = contract('ijk, jk,i->jk', Sxx, gx(qreal), pyp0.weights)
    SNSFzz= contract('ijk,jk,i->', Szz, gNSF(qreal, v), pyp0.weights)
    SNSFxx = contract('ijk,jk,i->', Sxx, gNSFx(qreal, v), pyp0.weights)
    Szz = contract('ijk,i->jk', Szz, pyp0.weights)
    Sxx = contract('ijk,i->jk', Sxx, pyp0.weights)

    return Szz, Szzglobal, SNSFzz, Sxx, Sxxglobal, SNSFxx

def graph_SSSF_pedantic(pyp0, K, v, rank, size):
    comm = MPI.COMM_WORLD
    nK = len(K)
    nb = nK / size

    leftK = int(rank * nb)
    rightK = int((rank + 1) * nb)
    currsizeK = rightK - leftK

    currK = K[leftK:rightK, :]

    sendtemp = np.zeros((currsizeK,4,4), dtype=np.float64)
    sendtemp1 = np.zeros((currsizeK,4,4), dtype=np.float64)
    sendtemp2 = np.zeros((currsizeK,4,4), dtype=np.float64)
    sendtemp3 = np.zeros((currsizeK,4,4), dtype=np.float64)
    sendtemp4 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp5 = np.zeros(currsizeK, dtype=np.float64)

    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None
    rectemp5 = None

    if rank == 0:
        rectemp = np.zeros((len(K),4,4), dtype=np.float64)
        rectemp1 = np.zeros((len(K),4,4), dtype=np.float64)
        rectemp2 = np.zeros((len(K),4,4), dtype=np.float64)
        rectemp3 = np.zeros((len(K),4,4), dtype=np.float64)
        rectemp4 = np.zeros(len(K), dtype=np.float64)
        rectemp5 = np.zeros(len(K), dtype=np.float64)

    for i in range(currsizeK):
        sendtemp[i], sendtemp1[i], sendtemp4[i], sendtemp2[i], sendtemp3[i],sendtemp5[i] = SSSF_core_pedantic(currK[i], v, pyp0)

    sendcounts = np.array(comm.gather(len(sendtemp)*16, 0))
    sendcounts1 = np.array(comm.gather(len(sendtemp1)*16, 0))
    sendcounts2 = np.array(comm.gather(len(sendtemp2)*16, 0))
    sendcounts3 = np.array(comm.gather(len(sendtemp3)*16, 0))
    sendcounts4 = np.array(comm.gather(len(sendtemp4), 0))
    sendcounts5 = np.array(comm.gather(len(sendtemp5), 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)

    return rectemp, rectemp1, rectemp4, rectemp2, rectemp3, rectemp5


def SSSF_core(q, v, pyp0):
    Ks = pyp0.pts
    Qs = Ks - q

    if (v==0).all():
        v = np.array([-1,1,0])/np.sqrt(2)
    else:
        v = v/np.linalg.norm(v)

    Spm, Spp = SpmSpp(Ks, Qs, q, pyp0)
    if pyp0.dominant == 0:
        Sxx = (np.real(Spm) + np.real(Spp)) / 2
        Szz = (np.real(Spm) - np.real(Spp)) / 2
    else:
        Szz = (np.real(Spm) + np.real(Spp)) / 2
        Sxx = (np.real(Spm) - np.real(Spp)) / 2

    qreal = contract('j,jk->k',q, BasisBZA)
    G, TV = gTransverse(qreal)
    Sglobalzz = contract('ijk,jk,i->', Szz, G, pyp0.weights)
    SglobalzzT = contract('ijk,jk,i->', Szz, TV, pyp0.weights)
    SNSFzz = contract('ijk,jk,i->', Szz, gNSF(qreal, v), pyp0.weights)
    Szz = contract('ijk,i->', Szz, pyp0.weights)
    Sglobalxx = contract('ijk,jk,i->', Sxx, G, pyp0.weights)
    SglobalxxT = contract('ijk,jk,i->', Sxx, TV, pyp0.weights)
    SNSFxx = contract('ijk,jk,i->', Sxx, gNSF(qreal, v), pyp0.weights)
    Sxx = contract('ijk,i->', Sxx, pyp0.weights)
    return Szz, Sglobalzz, SglobalzzT, SNSFzz, Sxx, Sglobalxx, SglobalxxT, SNSFxx

def graph_SSSF(pyp0, K, V, rank, size):
    comm = MPI.COMM_WORLD
    nK = len(K)
    nb = nK / size

    leftK = int(rank * nb)
    rightK = int((rank + 1) * nb)
    currsizeK = rightK - leftK

    currK = K[leftK:rightK, :]

    sendtemp = np.zeros(currsizeK, dtype=np.float64)
    sendtemp1 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp2 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp3 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp4 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp5 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp6 = np.zeros(currsizeK, dtype=np.float64)
    sendtemp7 = np.zeros(currsizeK, dtype=np.float64)

    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None
    rectemp5 = None
    rectemp6 = None
    rectemp7 = None

    if rank == 0:
        rectemp = np.zeros(len(K), dtype=np.float64)
        rectemp1 = np.zeros(len(K), dtype=np.float64)
        rectemp2 = np.zeros(len(K), dtype=np.float64)
        rectemp3 = np.zeros(len(K), dtype=np.float64)
        rectemp4 = np.zeros(len(K), dtype=np.float64)
        rectemp5 = np.zeros(len(K), dtype=np.float64)
        rectemp6 = np.zeros(len(K), dtype=np.float64)
        rectemp7 = np.zeros(len(K), dtype=np.float64)

    for i in range(currsizeK):
        sendtemp[i], sendtemp1[i], sendtemp2[i], sendtemp3[i], sendtemp4[i], sendtemp5[i], sendtemp6[i], sendtemp7[i] = SSSF_core(currK[i], V, pyp0)

    sendcounts = np.array(comm.gather(len(sendtemp), 0))
    sendcounts1 = np.array(comm.gather(len(sendtemp1), 0))
    sendcounts2 = np.array(comm.gather(len(sendtemp2), 0))
    sendcounts3 = np.array(comm.gather(len(sendtemp3), 0))
    sendcounts4 = np.array(comm.gather(len(sendtemp4), 0))
    sendcounts5 = np.array(comm.gather(len(sendtemp5), 0))
    sendcounts6 = np.array(comm.gather(len(sendtemp6), 0))
    sendcounts7 = np.array(comm.gather(len(sendtemp7), 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)
    comm.Gatherv(sendbuf=sendtemp6, recvbuf=(rectemp6, sendcounts6), root=0)
    comm.Gatherv(sendbuf=sendtemp7, recvbuf=(rectemp7, sendcounts7), root=0)

    return rectemp, rectemp1, rectemp2, rectemp3, rectemp4, rectemp5, rectemp6, rectemp7


# endregion

# region Graphing
def DSSFgraph(D, filename, A, B):
    plt.imshow(D, interpolation="lanczos",origin='lower',extent =[A.min(), A.max(), B.min(), B.max()], aspect='auto')
    plt.ylabel(r'$\omega/J_{yy}$')
    plt.savefig(filename + ".pdf")
    plt.clf()

def DSSFgraph_pedantic(D, filename, A, B, lowedge, upedge):
    plt.imshow(D.T, interpolation="lanczos",origin='lower',extent =[A.min(), A.max(), B.min(), B.max()], aspect='auto')


    plt.plot(A, lowedge, 'b',zorder=8)
    plt.plot(A, upedge, 'b',zorder=8)

    plt.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma3, color='w', label='axvline - full height', linestyle='dashed')

    xlabpos = [gGamma1, gX, gW, gK, gGamma2, gL, gU, gW1, gX1, gGamma3]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W^\prime$', r'$X^\prime$', r'$\Gamma$']
    plt.xticks(xlabpos, labels)
    plt.xlim([0,gGamma3])

    plt.ylabel(r'$\omega/J_{yy}$')
    plt.savefig(filename + ".pdf")
    plt.clf()

def plot_line(A, B, color):
    temp = np.array([A,B]).T
    plt.plot(temp[0], temp[1], color,zorder=5)

def plot_text(A, text):
    temp = A + 0.05*np.array([1,-1])
    plt.text(temp[0,0],temp[0,1],text)

def plot_BZ_hhl(offset, boundary, color):
    B = boundary+offset
    plot_line(B[0],B[1],color)
    plot_line(B[1],B[2],color)
    plot_line(B[2],B[3],color)
    plot_line(B[3],B[4],color)
    plot_line(B[4],B[5],color)
    plot_line(B[5],B[0],color)

def plot_BZ_hkk(offset, boundary, color):
    B = boundary+offset
    plot_line(B[0],B[1],color)
    plot_line(B[1],B[2],color)
    plot_line(B[2],B[3],color)
    plot_line(B[3],B[4],color)
    plot_line(B[4], B[5], color)
    plot_line(B[5], B[6], color)
    plot_line(B[6], B[7], color)
    plot_line(B[7], B[0], color)


def SSSFGraph_helper(d1, Hr, Lr, vmin, vmax):
    if np.isnan(vmin) or np.isnan(vmax):
        plt.imshow(d1, interpolation="lanczos", origin='lower', extent=[-Hr, Hr, -Lr, Lr], aspect='auto')
    else:
        plt.imshow(d1, interpolation="lanczos", origin='lower', extent=[-Hr, Hr, -Lr, Lr], aspect='auto', vmin=vmin,
                   vmax=vmax)
    plt.colorbar()

def SSSFGraphHKK(d1, filename, Hr, Lr, vmin=np.nan, vmax=np.nan):

    SSSFGraph_helper(d1, Hr, Lr, vmin, vmax)

    Gamms = np.array([[0,0],[1,1],[1,-1],[-1,1],[-1,-1],[2,0],[-2,0],[0,2],[0,-2],[2,2],[-2,2],[2,-2],[-2,-2]])

    Xs = np.array([[0.5,0.5]])
    Ws = np.array([[0.75,-0.25]])
    Ks = np.array([[0.75,0]])

    plot_text(Gamms,r'$\Gamma$')
    plot_text(Xs,r'$X$')
    plot_text(Ws,r'$W$')
    plot_text(Ks,r'$K$')



    Boundary = np.array([[0.75, -0.25],[0.25,-0.75],[-0.25,-0.75],[-0.75,-0.25],[-0.75,0.25],[-0.25,0.75],[0.25,0.75],[0.75,0.25]])
    # Boundary = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    plt.scatter(Gamms[0,0],Gamms[0,1],zorder=6)
    plt.scatter(Xs[:,0], Xs[:,1],zorder=6)
    plt.scatter(Ks[:,0], Ks[:,1],zorder=6)
    plt.scatter(Ws[:, 0], Ws[:, 1],zorder=6)

    plot_BZ_hkk(Gamms[0], Boundary, 'w:')


    plt.ylabel(r'$(K,-K,0)$')
    plt.xlabel(r'$(H,H,0)$')
    plt.xlim([-Hr, Hr])
    plt.ylim([-Lr, Lr])
    # plt.show()
    plt.savefig(filename + ".pdf")
    plt.clf()

def SSSFGraphHHL(d1, filename, Hr, Lr, vmin=np.nan, vmax=np.nan):
    SSSFGraph_helper(d1, Hr, Lr, vmin, vmax)

    Gamms = np.array([[0,0],[1,1],[-1,1],[1,-1],[-1,-1],[2,0],[0,2],[-2,0],[0,-2],[2,2],[-2,2],[2,-2],[-2,-2]])
    Ls = np.array([[0.5,0.5]])
    Xs = np.array([[0,1]])
    Us = np.array([[0.25,1]])
    Ks = np.array([[0.75,0]])


    Boundary = np.array([[0.25, 1],[-0.25,1],[-0.75,0],[-0.25,-1],[0.25,-1],[0.75,0]])



    plot_BZ_hhl(Gamms[0], Boundary, 'w:')

    plt.scatter(Gamms[0,0], Gamms[0,1],zorder=6)
    plt.scatter(Ls[:,0], Ls[:,1],zorder=6)
    plt.scatter(Xs[:, 0], Xs[:, 1],zorder=6)
    plt.scatter(Ks[:, 0], Ks[:, 1],zorder=6)
    plt.scatter(Us[:, 0], Us[:, 1],zorder=6)
    plot_text(Gamms,r'$\Gamma$')
    plot_text(Ls,r'$L$')
    plot_text(Xs,r'$X$')
    plot_text(Us,r'$U$')
    plot_text(Ks,r'$K$')

    plt.ylabel(r'$(0,0,L)$')
    plt.xlabel(r'$(H,-H,0)$')
    plt.xlim([-Hr, Hr])
    plt.ylim([-Lr, Lr])
    # plt.show()
    plt.savefig(filename + ".pdf")
    plt.clf()
def SSSFGraphHK0(d1, filename, Hr, Lr, vmin=np.nan, vmax=np.nan):
    SSSFGraph_helper(d1, Hr, Lr, vmin, vmax)

    #
    # Gamms = np.array([[0,0],[2,0],[0,2],[-2,0],[0,-2],[2,2],[-2,2],[2,-2],[-2,-2]])
    # Xs = np.array([[1, 0]])
    # Ks = np.array([[1,1]])*0.375/0.5
    # Ws = np.array([[1,0.5]])
    #
    # Boundary = np.array([[1, 0.5], [0.5,1], [-0.5,1], [-1,0.5],[-1,-0.5],[-0.5,-1],[0.5,-1],[1,-0.5]])
    #
    # plt.scatter(Gamms[0,0], Gamms[0,1],zorder=6)
    # plt.scatter(Xs[:, 0], Xs[:, 1], zorder=6)
    # plt.scatter(Ks[:, 0], Ks[:, 1], zorder=6)
    # plt.scatter(Ws[:, 0], Ws[:, 1], zorder=6)
    # plot_text(Gamms,r'$\Gamma$')
    # plot_text(Xs,r'$X$')
    # plot_text(Ks,r'$K$')
    # plot_text(Ws,r'$W$')
    #
    # plot_BZ_hkk(Gamms[0], Boundary, 'b:')

    plt.ylabel(r'$(0,K,0)$')
    plt.xlabel(r'$(H,0,0)$')
    plt.xlim([-Hr, Hr])
    plt.ylim([-Lr, Lr])
    plt.savefig(filename + ".pdf")
    plt.clf()


def SSSFGraphHH2K(d1, filename, Hr, Lr, vmin=np.nan, vmax=np.nan):
    SSSFGraph_helper(d1, Hr, Lr, vmin, vmax)
    plt.ylabel(r'$(K,K,-2K)$')
    plt.xlabel(r'$(H,-H,0)$')
    plt.xlim([-Hr, Hr])
    plt.ylim([-Lr, Lr])
    plt.savefig(filename + ".pdf")
    plt.clf()

def SSSFGraphHnHL(d1, filename, Hr, Lr, vmin=np.nan, vmax=np.nan):
    SSSFGraph_helper(d1, Hr, Lr, vmin, vmax)
    plt.ylabel(r'$(0,0, L)$')
    plt.xlabel(r'$(H,-H,0)$')
    plt.xlim([-Hr, Hr])
    plt.ylim([-Lr, Lr])
    plt.savefig(filename + ".pdf")
    plt.clf()




# endregion


# region SSSF DSSF Admin

def SSSF_Ks(K, Jxx, Jyy, Jzz, h, n, flux, BZres, filename):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()

    scatterPlane = q_scaplane(K)
    v = np.zeros(scatterPlane.shape)
    v[:, 0] = -scatterPlane[:, 1]
    v[:, 1] = scatterPlane[:, 0]

    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1, d2, d3, d4, d5, d6, d7, d8 = graph_SSSF(py0s, K, v, rank, size)
    if rank == 0:
        f1 = filename + "Szz_local"
        f2 = filename + "Szz_global"
        f3 = filename + "Szz_globalT"
        f4 = filename + "Szz_NSF"
        f5 = filename + "Sxx_local"
        f6 = filename + "Sxx_global"
        f7 = filename + "Sxx_globalT"
        f8 = filename + "Sxx_NSF"
        np.savetxt(f1 + '.txt', d1)
        np.savetxt(f2 + '.txt', d2)
        np.savetxt(f3 + '.txt', d3)
        np.savetxt(f4 + '.txt', d4)
        np.savetxt(f5 + '.txt', d5)
        np.savetxt(f6 + '.txt', d6)
        np.savetxt(f7 + '.txt', d7)
        np.savetxt(f8 + '.txt', d8)

def pedantic_SSSF_graph_helper(graphMethod, d1, f1, Hr, Lr, dir):
    for i in range(4):
        for j in range(4):
            tempF = f1+str(i)+str(j)
            np.savetxt(tempF + '.txt', d1[:,:,i,j])
            graphMethod(d1[:,:,i,j], tempF, Hr, Lr)
    if (dir==h110).all():
        gp = d1[:,:,0,0] + d1[:,:,0,3] + d1[:,:,3,0] + d1[:,:,3,3] 
        gup = d1[:,:,1,1] + d1[:,:,1,2] + d1[:,:,2,1] + d1[:,:,2,2]
        gcorre = d1[:,:,0,1] + d1[:,:,0,2] + d1[:,:,1,0] + d1[:,:,2,0] + d1[:,:,3,1] + d1[:,:,3,2] + d1[:,:,1,3] + d1[:,:,2,3]
        np.savetxt(f1+"polarized.txt", gp)
        np.savetxt(f1+"unpolarized.txt", gup)
        np.savetxt(f1+"polar_unpolar.txt", gcorre)
        graphMethod(gp, f1+"polarized", Hr, Lr)
        graphMethod(gup, f1+"unpolarized", Hr, Lr)
        graphMethod(gcorre, f1+"polar_unpolar", Hr, Lr)
    elif (dir==h111).all():
        gKagome = d1[:,:,1,1] + d1[:,:,1,2] + d1[:,:,1,3] + d1[:,:,2,1] + d1[:,:,3, 1] + d1[:,:,2,2] + d1[:,:,2,3] + d1[:,:,3,2] + d1[:,:,3,3]
        gTri = d1[:,:,0,0]
        gKagomeTri = d1[:,:,0,1] + d1[:,:,0,2] + d1[:,:,0,3] + d1[:,:,1,0] + d1[:,:,2,0] + d1[:,:,3,0]
        np.savetxt(f1+"Kagome.txt", gKagome)
        np.savetxt(f1+"Triangular.txt", gTri)
        np.savetxt(f1+"Kagome-Tri.txt", gKagomeTri)
        graphMethod(gKagome, f1+"Kagome", Hr, Lr)
        graphMethod(gTri, f1+"Triangular", Hr, Lr)
        graphMethod(gKagomeTri, f1+"Kagome-Tri", Hr, Lr)
    else:
        gp = d1[:,:,0,0] + d1[:,:,0,3] + d1[:,:,3,0] + d1[:,:,3,3]
        gup = d1[:,:,1,1] + d1[:,:,1,2] + d1[:,:,2,1] + d1[:,:,2,2]
        gcorre = d1[:,:,0,1] + d1[:,:,0,2] + d1[:,:,1,0] + d1[:,:,2,0] + d1[:,:,3,1] + d1[:,:,3,2] + d1[:,:,1,3] + d1[:,:,2,3]
        np.savetxt(f1+"polarized.txt", gp)
        np.savetxt(f1+"unpolarized.txt", gup)
        np.savetxt(f1+"polar_unpolar.txt", gcorre)
        graphMethod(gp, f1+"polarized", Hr, Lr)
        graphMethod(gup, f1+"unpolarized", Hr, Lr)
        graphMethod(gcorre, f1+"polar_unpolar", Hr, Lr)

def SSSF_pedantic(nK, Jxx, Jyy, Jzz, h, n, flux, BZres, filename, hkl, *args, K=0, Hr=2.5, Lr=2.5, g=0, theta=0):
    pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, *args, BZres=BZres, h=h, n=n, flux=flux, theta=theta)
    py0s.solvemeanfield()
    H = np.linspace(-Hr, Hr, nK)
    L = np.linspace(-Lr, Lr, nK)
    A, B = np.meshgrid(H, L)

    if hkl == "hk0":
        K = hkztoK(A, B, K).reshape((nK*nK,3))
    elif hkl=="hnhl":
        K = hnhltoK(A, B, K).reshape((nK * nK, 3))
    elif hkl=="hhl":
        K = hhltoK(A, B, K).reshape((nK * nK, 3))
    elif hkl=="hhknk":
        K = hhknktoK(A, B, K).reshape((nK * nK, 3))
    else:
        K = hnhkkn2ktoK(A, B, K).reshape((nK * nK, 3))

    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1, d2, d5, d3, d4, d6 = graph_SSSF_pedantic(py0s, K, n, rank, size)
    if rank == 0:
        pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
        f1 = filename + "/Szz/"
        f2 = filename + "/Szzglobal/"
        f3 = filename + "/Sxx/"
        f4 = filename + "/Sxxglobal/"
        pathlib.Path(f1).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f2).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f3).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f4).mkdir(parents=True, exist_ok=True)

        d1 = d1.reshape((nK, nK, 4, 4))
        d2 = d2.reshape((nK, nK, 4, 4))
        d3 = d3.reshape((nK, nK, 4, 4))
        d4 = d4.reshape((nK, nK, 4, 4))
        d5 = d5.reshape((nK, nK))
        d6 = d6.reshape((nK, nK))

        Szz = contract('abjk->ab', d1)
        Szzglobal = contract('abjk->ab', d2)
        Sxx = contract('abjk->ab', d3)
        Sxxglobal = contract('abjk->ab', d4)


        if hkl=="hk0":
            np.savetxt(filename+'/Szz' + '.txt', Szz)
            np.savetxt(filename+'/Szzglobal' + '.txt', Szzglobal)
            np.savetxt(filename+'/SzzNSF' + '.txt', d5)
            np.savetxt(filename+'/Sxx' + '.txt', Sxx)
            np.savetxt(filename+'/Sxxglobal' + '.txt', Sxxglobal)
            np.savetxt(filename+'/SxxNSF' + '.txt', d6)
            SSSFGraphHK0(Szz, filename+'/Szz', Hr, Lr)
            SSSFGraphHK0(Szzglobal, filename + '/Szzglobal', Hr, Lr)
            SSSFGraphHK0(Sxx, filename + '/Sxx', Hr, Lr)
            SSSFGraphHK0(Sxxglobal, filename + '/Sxxglobal', Hr, Lr)
            SSSFGraphHK0(d5, filename + '/SzzNSF', Hr, Lr)
            SSSFGraphHK0(d6, filename + '/SxxNSF', Hr, Lr)
            pedantic_SSSF_graph_helper(SSSFGraphHK0, d1, f1, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHK0, d2, f2, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHK0, d3, f3, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHK0, d4, f4, Hr, Lr, n)            
        elif hkl=="hhl":
            np.savetxt(filename+'/Szz' + '.txt', Szz)
            np.savetxt(filename+'/Szzglobal' + '.txt', Szzglobal)
            np.savetxt(filename+'/SzzNSF' + '.txt', d5)
            np.savetxt(filename+'/Sxx' + '.txt', Sxx)
            np.savetxt(filename+'/Sxxglobal' + '.txt', Sxxglobal)
            np.savetxt(filename+'/SxxNSF' + '.txt', d6)
            SSSFGraphHHL(Szz, filename+'/Szz', Hr, Lr)
            SSSFGraphHHL(Szzglobal, filename + '/Szzglobal', Hr, Lr)
            SSSFGraphHHL(Sxx, filename + '/Sxx', Hr, Lr)
            SSSFGraphHHL(Sxxglobal, filename + '/Sxxglobal', Hr, Lr)
            SSSFGraphHHL(d5, filename + '/SzzNSF', Hr, Lr)
            SSSFGraphHHL(d6, filename + '/SxxNSF', Hr, Lr)
            pedantic_SSSF_graph_helper(SSSFGraphHHL, d1, f1, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHHL, d2, f2, Hr, Lr, n)            
            pedantic_SSSF_graph_helper(SSSFGraphHHL, d3, f3, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHHL, d4, f4, Hr, Lr, n)
        elif hkl=="hkk":
            np.savetxt(filename+'/Szz' + '.txt', Szz)
            np.savetxt(filename+'/Szzglobal' + '.txt', Szzglobal)
            np.savetxt(filename+'/SzzNSF' + '.txt', d5)
            np.savetxt(filename+'/Sxx' + '.txt', Sxx)
            np.savetxt(filename+'/Sxxglobal' + '.txt', Sxxglobal)
            np.savetxt(filename+'/SxxNSF' + '.txt', d6)
            SSSFGraphHKK(Szz, filename+'/Szz', Hr, Lr)
            SSSFGraphHKK(Szzglobal, filename + '/Szzglobal', Hr, Lr)
            SSSFGraphHKK(Sxx, filename + '/Sxx', Hr, Lr)
            SSSFGraphHKK(Sxxglobal, filename + '/Sxxglobal', Hr, Lr)
            SSSFGraphHKK(d5, filename + '/SzzNSF', Hr, Lr)
            SSSFGraphHKK(d6, filename + '/SxxNSF', Hr, Lr)
            pedantic_SSSF_graph_helper(SSSFGraphHKK, d1, f1, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHKK, d2, f2, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHKK, d3, f3, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHKK, d4, f4, Hr, Lr, n)
        elif hkl=="hnhl":
            np.savetxt(filename+'/Szz' + '.txt', Szz)
            np.savetxt(filename+'/Szzglobal' + '.txt', Szzglobal)
            np.savetxt(filename+'/SzzNSF' + '.txt', d5)
            np.savetxt(filename+'/Sxx' + '.txt', Sxx)
            np.savetxt(filename+'/Sxxglobal' + '.txt', Sxxglobal)
            np.savetxt(filename+'/SxxNSF' + '.txt', d6)
            SSSFGraphHnHL(Szz, filename+'/Szz', Hr, Lr)
            SSSFGraphHnHL(Szzglobal, filename + '/Szzglobal', Hr, Lr)
            SSSFGraphHnHL(Sxx, filename + '/Sxx', Hr, Lr)
            SSSFGraphHnHL(Sxxglobal, filename + '/Sxxglobal', Hr, Lr)
            SSSFGraphHnHL(d5, filename + '/SzzNSF', Hr, Lr)
            SSSFGraphHnHL(d6, filename + '/SxxNSF', Hr, Lr)
            pedantic_SSSF_graph_helper(SSSFGraphHnHL, d1, f1, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHnHL, d2, f2, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHnHL, d3, f3, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHnHL, d4, f4, Hr, Lr, n)
        else:
            np.savetxt(filename+'/Szz' + '.txt', Szz)
            np.savetxt(filename+'/Szzglobal' + '.txt', Szzglobal)
            np.savetxt(filename+'/SzzNSF' + '.txt', d5)
            np.savetxt(filename+'/Sxx' + '.txt', Sxx)
            np.savetxt(filename+'/Sxxglobal' + '.txt', Sxxglobal)
            np.savetxt(filename+'/SxxNSF' + '.txt', d6)
            SSSFGraphHH2K(Szz, filename+'/Szz', Hr, Lr)
            SSSFGraphHH2K(Szzglobal, filename + '/Szzglobal', Hr, Lr)
            SSSFGraphHH2K(Sxx, filename + '/Sxx', Hr, Lr)
            SSSFGraphHH2K(Sxxglobal, filename + '/Sxxglobal', Hr, Lr)
            SSSFGraphHH2K(d5, filename + '/SzzNSF', Hr, Lr)
            SSSFGraphHH2K(d6, filename + '/SxxNSF', Hr, Lr)
            pedantic_SSSF_graph_helper(SSSFGraphHH2K, d1, f1, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHH2K, d2, f2, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHH2K, d3, f3, Hr, Lr, n)
            pedantic_SSSF_graph_helper(SSSFGraphHH2K, d4, f4, Hr, Lr, n)

def SSSF_BZ(nK, Jxx, Jyy, Jzz, h, n, flux, BZres, filename, theta):
    pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux, theta=theta)
    py0s.solvemeanfield()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1, d2, d5, d3, d4, d6 = graph_SSSF_pedantic(py0s, py0s.pts, n, rank, size)
    if rank == 0:
        pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
        f1 = filename + "/Szz/"
        f2 = filename + "/Szzglobal/"
        f3 = filename + "/Sxx/"
        f4 = filename + "/Sxxglobal/"
        pathlib.Path(f1).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f2).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f3).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f4).mkdir(parents=True, exist_ok=True)

        # d1 = d1.reshape((nK, nK, 4, 4))
        # d2 = d2.reshape((nK, nK, 4, 4))
        # d3 = d3.reshape((nK, nK, 4, 4))
        # d4 = d4.reshape((nK, nK, 4, 4))
        # d5 = d5.reshape((nK, nK))
        # d6 = d6.reshape((nK, nK))

        Szz = contract('ajk, a->', d1, py0s.weights)
        Sxx = contract('ajk, a->', d3, py0s.weights)
        Q_sum = contract('a->', py0s.weights)
        np.savetxt(filename+'/S' + '.txt', np.array([Szz, Sxx])/Q_sum)    

def SSSF_q_omega_beta(beta, nK, nE, Jxx, Jyy, Jzz, h, n, flux, BZres, filename, hkl, *args, K=0, Hr=2.5, Lr=2.5, g=0, theta=0):
    pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, *args, BZres=BZres, h=h, n=n, flux=flux, theta=theta)
    py0s.solvemeanfield()
    H = np.linspace(-Hr, Hr, nK)
    L = np.linspace(-Lr, Lr, nK)
    A, B = np.meshgrid(H, L)

    if hkl == "hk0":
        K = hkztoK(A, B, K).reshape((nK*nK,3))
    elif hkl=="hnhl":
        K = hnhltoK(A, B, K).reshape((nK * nK, 3))
    elif hkl=="hhl":
        K = hhltoK(A, B, K).reshape((nK * nK, 3))
    elif hkl=="hhknk":
        K = hhknktoK(A, B, K).reshape((nK * nK, 3))
    else:
        K = hnhkkn2ktoK(A, B, K).reshape((nK * nK, 3))

    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # emin, emax = np.min(py0s.graph_loweredge(False)), np.max(py0s.graph_upperedge(False))
    # e = np.linspace(max(emin *0.95, 0), emax *1.02, nE)
    e = np.linspace(0, 10, nE)
    tol = 1/nE*4
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1, d2, d3, d4 = graph_DSSF_pedantic(py0s, e, K, tol, rank, size, beta)

    if rank == 0:
        d1 = contract('ijkl->ij', d1)
        d2 = contract('ijkl->ij', d2)
        d3 = contract('ijkl->ij', d3)
        d4 = contract('ijkl->ij', d4)

        f1 = filename + "_Szz_local"
        f2 = filename + "_Szz_global"
        f3 = filename + "_Sxx_local"
        f4 = filename + "_Sxx_global"
        np.savetxt(f1 + ".txt", d1)
        np.savetxt(f2 + ".txt", d2)
        np.savetxt(f3 + ".txt", d3)
        np.savetxt(f4 + ".txt", d4)

def SSSF_q_omega_beta_at_K(beta, K, nE, Jxx, Jyy, Jzz, h, n, flux, BZres, theta):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux, theta=theta)
    py0s.solvemeanfield()


    # emin, emax = np.min(py0s.graph_loweredge(False)), np.max(py0s.graph_upperedge(False))
    # e = np.linspace(max(emin *0.95, 0), emax *1.02, nE)
    e = np.linspace(0, 10, nE)
    tol = 1/nE*4
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1, d2, d3, d4 = graph_DSSF_pedantic(py0s, e, K, tol, rank, size, beta)

    return contract('ijab->ij',d1), contract('ijab->ij',d3)


def SSSF(nK, Jxx, Jyy, Jzz, h, n, flux, BZres, filename, hkl, K=0, Hr=2.5, Lr=2.5):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    H = np.linspace(-Hr, Hr, nK)
    L = np.linspace(-Lr, Lr, nK)
    A, B = np.meshgrid(H, L)

    if hkl == "hk0":
        K = hkztoK(A, B, K).reshape((nK*nK,3))
    elif hkl=="hnhl":
        K = hnhltoK(A, B, K).reshape((nK * nK, 3))
    elif hkl=="hhl":
        K = hhltoK(A, B, K).reshape((nK * nK, 3))
    elif hkl=="hhknk":
        K = hhknktoK(A, B, K).reshape((nK * nK, 3))
    else:
        K = hnhkkn2ktoK(A, B, K).reshape((nK * nK, 3))

    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1, d2, d3, d4, d5, d6, d7, d8 = graph_SSSF(py0s, K, n, rank, size)
    if rank == 0:
        f1 = filename + "Szz_local"
        f2 = filename + "Szz_global"
        f3 = filename + "Szz_globalT"
        f4 = filename + "Szz_NSF"
        f5 = filename + "Sxx_local"
        f6 = filename + "Sxx_global"
        f7 = filename + "Sxx_globalT"
        f8 = filename + "Sxx_NSF"
        d1 = d1.reshape((nK, nK))
        d2 = d2.reshape((nK, nK))
        d3 = d3.reshape((nK, nK))
        d4 = d4.reshape((nK, nK))
        d5 = d5.reshape((nK, nK))
        d6 = d6.reshape((nK, nK))
        d7 = d7.reshape((nK, nK))
        d8 = d8.reshape((nK, nK))
        np.savetxt(f1 + '.txt', d1)
        np.savetxt(f2 + '.txt', d2)
        np.savetxt(f3 + '.txt', d3)
        np.savetxt(f4 + '.txt', d4)
        np.savetxt(f5 + '.txt', d5)
        np.savetxt(f6 + '.txt', d6)
        np.savetxt(f7 + '.txt', d7)
        np.savetxt(f8 + '.txt', d8)
        if hkl=="hk0":
            SSSFGraphHK0(d1, f1, Hr, Lr)
            SSSFGraphHK0(d2, f2, Hr, Lr)
            SSSFGraphHK0(d3, f3, Hr, Lr)
            SSSFGraphHK0(d4, f4, Hr, Lr)
            SSSFGraphHK0(d5, f5, Hr, Lr)
            SSSFGraphHK0(d6, f6, Hr, Lr)
            SSSFGraphHK0(d7, f7, Hr, Lr)
            SSSFGraphHK0(d8, f8, Hr, Lr)
        elif hkl=="hhl":
            SSSFGraphHHL(d1, f1, Hr, Lr)
            SSSFGraphHHL(d2, f2, Hr, Lr)
            SSSFGraphHHL(d3, f3, Hr, Lr)
            SSSFGraphHHL(d4, f4, Hr, Lr)
            SSSFGraphHHL(d5, f5, Hr, Lr)
            SSSFGraphHHL(d6, f6, Hr, Lr)
            SSSFGraphHHL(d7, f7, Hr, Lr)
            SSSFGraphHHL(d8, f8, Hr, Lr)
        else:
            SSSFGraphHKK(d1, f1, Hr, Lr)
            SSSFGraphHKK(d2, f2, Hr, Lr)
            SSSFGraphHKK(d3, f3, Hr, Lr)
            SSSFGraphHKK(d4, f4, Hr, Lr)
            SSSFGraphHKK(d5, f5, Hr, Lr)
            SSSFGraphHKK(d6, f6, Hr, Lr)
            SSSFGraphHKK(d7, f7, Hr, Lr)
            SSSFGraphHKK(d8, f8, Hr, Lr)


def SSSF_HHL_KK_integrated(nK, Jxx, Jyy, Jzz, h, n, flux, Lmin, Lmax, Ln, BZres, filename, Hr=2.5, Lr=2.5):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    H = np.linspace(-Hr, Hr, nK)
    L = np.linspace(-Lr, Lr, nK)
    K, Kweight = gauss_quadrature_1D_pts(Lmin,Lmax,Ln)

    A, B = np.meshgrid(H, L)

    Q = hhlK(A, K, B)


    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    d1 = np.zeros((Ln, nK * nK))
    d2 = np.zeros((Ln, nK * nK))
    d3 = np.zeros((Ln, nK * nK))
    d4 = np.zeros((Ln, nK * nK))
    d5 = np.zeros((Ln, nK * nK))
    d6 = np.zeros((Ln, nK * nK))

    for i in range(Ln):
        d1[i], d2[i], d3[i], d4[i], d5[i], d6[i] = graph_SSSF(py0s, Q[i], np.einsum('k,i->ik',hb110,np.ones(nK*nK)), rank, size)

    d1 = contract('li,l->i',d1,Kweight)
    d2 = contract('li,l->i',d2,Kweight)
    d3 = contract('li,l->i',d3,Kweight)
    d4 = contract('li,l->i',d4,Kweight)
    d5 = contract('li,l->i',d5,Kweight)
    d6 = contract('li,l->i',d6,Kweight)
    if rank == 0:
        f1 = filename + "Szz_local"
        f2 = filename + "Szz_global"
        f3 = filename + "Szz_NSF"
        f4 = filename + "Sxx_local"
        f5 = filename + "Sxx_global"
        f6 = filename + "Sxx_NSF"
        d1 = d1.reshape((nK, nK))
        d2 = d2.reshape((nK, nK))
        d3 = d3.reshape((nK, nK))
        d4 = d4.reshape((nK, nK))
        d5 = d5.reshape((nK, nK))
        d6 = d6.reshape((nK, nK))
        np.savetxt(f1 + '.txt', d1)
        np.savetxt(f2 + '.txt', d2)
        np.savetxt(f3 + '.txt', d3)
        np.savetxt(f4 + '.txt', d4)
        np.savetxt(f5 + '.txt', d5)
        np.savetxt(f6 + '.txt', d6)
        SSSFGraphHHL(d1, f1, Hr, Lr)
        SSSFGraphHHL(d2, f2, Hr, Lr)
        SSSFGraphHHL(d3, f3, Hr, Lr)
        SSSFGraphHHL(d4, f4, Hr, Lr)
        SSSFGraphHHL(d5, f5, Hr, Lr)
        SSSFGraphHHL(d6, f6, Hr, Lr)

def SSSF_HHKnK_L_integrated(nK, Jxx, Jyy, Jzz, h, n, flux, Lmin, Lmax, Ln, BZres, filename, Hr=2.5, Lr=2.5):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    H = np.linspace(-Hr, Hr, nK)
    K = np.linspace(-Lr, Lr, nK)
    L, Lweight = gauss_quadrature_1D_pts(Lmin,Lmax,Ln)

    A, B = np.meshgrid(H, K)

    Q = hknkL(A, B, L) #K with L index


    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    d1 = np.zeros((Ln, nK * nK))
    d2 = np.zeros((Ln, nK * nK))
    d3 = np.zeros((Ln, nK * nK))
    d4 = np.zeros((Ln, nK * nK))
    d5 = np.zeros((Ln, nK * nK))
    d6 = np.zeros((Ln, nK * nK))

    for i in range(Ln):
        d1[i], d2[i], d3[i], d4[i], d5[i], d6[i] = graph_SSSF(py0s, Q[i], np.einsum('k,i->ik',hb110,np.ones(nK*nK)), rank, size)

    d1 = contract('li,l->i',d1,Lweight)
    d2 = contract('li,l->i',d2,Lweight)
    d3 = contract('li,l->i',d3,Lweight)
    d4 = contract('li,l->i',d4,Lweight)
    d5 = contract('li,l->i',d5,Lweight)
    d6 = contract('li,l->i',d6,Lweight)
    if rank == 0:
        f1 = filename + "Szz_local"
        f2 = filename + "Szz_global"
        f3 = filename + "Szz_NSF"
        f4 = filename + "Sxx_local"
        f5 = filename + "Sxx_global"
        f6 = filename + "Sxx_NSF"
        d1 = d1.reshape((nK, nK))
        d2 = d2.reshape((nK, nK))
        d3 = d3.reshape((nK, nK))
        d4 = d4.reshape((nK, nK))
        d5 = d5.reshape((nK, nK))
        d6 = d6.reshape((nK, nK))
        np.savetxt(f1 + '.txt', d1)
        np.savetxt(f2 + '.txt', d2)
        np.savetxt(f3 + '.txt', d3)
        np.savetxt(f4 + '.txt', d4)
        np.savetxt(f5 + '.txt', d5)
        np.savetxt(f6 + '.txt', d6)
        SSSFGraphHKK(d1, f1, Hr, Lr)
        SSSFGraphHKK(d2, f2, Hr, Lr)
        SSSFGraphHKK(d3, f3, Hr, Lr)
        SSSFGraphHKK(d4, f4, Hr, Lr)
        SSSFGraphHKK(d5, f5, Hr, Lr)
        SSSFGraphHKK(d6, f6, Hr, Lr)

def SSSF_HK0_L_integrated(nK, Jxx, Jyy, Jzz, h, n, flux, Lmin, Lmax, Ln, BZres, filename, Hr=2.5, Lr=2.5):

    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    H = np.linspace(-Hr, Hr, nK)
    K = np.linspace(-Lr, Lr, nK)
    L, Lweight = gauss_quadrature_1D_pts(Lmin,Lmax,Ln)

    A, B = np.meshgrid(H, K)

    Q = hk0L(A, B, L)


    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    d1 = np.zeros((Ln, nK * nK))
    d2 = np.zeros((Ln, nK * nK))
    d3 = np.zeros((Ln, nK * nK))
    d4 = np.zeros((Ln, nK * nK))
    d5 = np.zeros((Ln, nK * nK))
    d6 = np.zeros((Ln, nK * nK))

    for i in range(Ln):
        d1[i], d2[i], d3[i], d4[i], d5[i], d6[i] = graph_SSSF(py0s, Q[i], np.einsum('k,i->ik',hb110,np.ones(nK*nK)), rank, size)

    d1 = contract('li,l->i',d1,Lweight)
    d2 = contract('li,l->i',d2,Lweight)
    d3 = contract('li,l->i',d3,Lweight)
    d4 = contract('li,l->i',d4,Lweight)
    d5 = contract('li,l->i',d5,Lweight)
    d6 = contract('li,l->i',d6,Lweight)
    if rank == 0:
        f1 = filename + "Szz_local"
        f2 = filename + "Szz_global"
        f3 = filename + "Szz_NSF"
        f4 = filename + "Sxx_local"
        f5 = filename + "Sxx_global"
        f6 = filename + "Sxx_NSF"
        d1 = d1.reshape((nK, nK))
        d2 = d2.reshape((nK, nK))
        d3 = d3.reshape((nK, nK))
        d4 = d4.reshape((nK, nK))
        d5 = d5.reshape((nK, nK))
        d6 = d6.reshape((nK, nK))
        np.savetxt(f1 + '.txt', d1)
        np.savetxt(f2 + '.txt', d2)
        np.savetxt(f3 + '.txt', d3)
        np.savetxt(f4 + '.txt', d4)
        np.savetxt(f5 + '.txt', d5)
        np.savetxt(f6 + '.txt', d6)
        SSSFGraphHK0(d1, f1, Hr, Lr)
        SSSFGraphHK0(d2, f2, Hr, Lr)
        SSSFGraphHK0(d3, f3, Hr, Lr)
        SSSFGraphHK0(d4, f4, Hr, Lr)
        SSSFGraphHK0(d5, f5, Hr, Lr)
        SSSFGraphHK0(d6, f6, Hr, Lr)
def DSSF(nE, Jxx, Jyy, Jzz, h, n, flux, BZres, filename):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    kk = np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW1, W1X1, X1Gamma))
    emin, emax = py0s.graph_loweredge(False), py0s.graph_upperedge(False)
    e = np.linspace(max(emin *0.95, 0), emax *1.02, nE)
    tol = (1.02*emax-max(emin *0.95, 0))/nE*4
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1, d2, d3, d4 = graph_DSSF(py0s, e, kk, tol, rank, size)

    if rank == 0:
        f1 = filename + "_Szz_local"
        f2 = filename + "_Szz_global"
        f3 = filename + "_Sxx_local"
        f4 = filename + "_Sxx_global"
        np.savetxt(f1 + ".txt", d1)
        np.savetxt(f2 + ".txt", d2)
        np.savetxt(f3 + ".txt", d3)
        np.savetxt(f4 + ".txt", d4)
        kline = np.concatenate((graphGammaX, graphXW, graphWK, graphKGamma, graphGammaL, graphLU, graphUW1, graphW1X1, graphX1Gamma))
        X, Y = np.meshgrid(kline, e)
        DSSFgraph(d1.T, f1, X, Y)
        py0s.graph_loweredge(False)
        py0s.graph_upperedge(False)
        DSSFgraph(d2.T, f2, X, Y)
        py0s.graph_loweredge(False)
        py0s.graph_upperedge(False)
        DSSFgraph(d3.T, f3, X, Y)
        py0s.graph_loweredge(False)
        py0s.graph_upperedge(False)
        DSSFgraph(d4.T, f4, X, Y)


def TwoSpinonDOS(emin, emax, nE, Jpm, h, n, flux, BZres, filename, Jpmpm=0):
    e = np.linspace(emin, emax, nE)
    tol = (emax-emin)/nE*4
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1 = graph_2S_rho(e, Jpm, h, n, flux, BZres, tol, rank, size)

    if rank == 0:
        f1 = filename + "two_spinon_DOS"
        np.savetxt(f1 + ".txt", d1)
        if isinstance(Jpm, np.ndarray):
            X, Y = np.meshgrid(Jpm, e)
            DSSFgraph(d1.T, f1, X, Y)
        else:
            X, Y = np.meshgrid(h, e)
            DSSFgraph(d1.T, f1, X, Y)

def TwoSpinonDOS_111(nH, BZres, filename, Jpmpm=0):
    if not MPI.Is_initialized():
        MPI.Init()
    E = np.linspace(0.2, 0.9, 200)
    tol = 1/200
    h = np.linspace(0,0.5,nH)
    Jpm=-0.1
    hn = h111
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1 = graph_2S_rho(E, Jpm, h, hn, BZres, rank, size, tol, Jpmpm)

    if rank == 0:
        f1 = filename
        np.savetxt(f1 + ".txt", d1)
        plt.imshow(d1.T, interpolation="lanczos", origin='lower', extent=[0, 0.5, 0.2, 0.9], aspect='auto', cmap='gnuplot')
        plt.ylabel(r'$\omega/J_{yy}$')
        plt.xlabel(r'$h/J_{yy}$')
        plt.savefig(filename)
        plt.clf()
        # DSSFgraph(d1.T, f1, X, Y)
def TwoSpinonDOS_001(nH, BZres, filename):
    if not MPI.Is_initialized():
        MPI.Init()
    E = np.linspace(0.3, 0.8, 200)
    tol = 1/200
    h = np.linspace(0.0,0.22,nH)
    Jpm=-0.03
    hn = h001
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1 = graph_2S_rho(E, Jpm, h, hn, BZres, rank, size, tol)

    if rank == 0:
        f1 = filename
        np.savetxt(f1 + ".txt", d1)
        plt.imshow(d1.T, interpolation="lanczos", origin='lower', extent=[0, 0.22, 0, 2], aspect='auto', cmap='gnuplot')
        plt.ylabel(r'$\omega/J_{yy}$')
        plt.xlabel(r'$h/J_{yy}$')
        plt.savefig(filename)
        plt.clf()
def TwoSpinonDOS_110(nH, BZres, filename):
    if not MPI.Is_initialized():
        MPI.Init()
    E = np.linspace(0.22, 0.8, 200)
    tol = 1/200
    h = np.linspace(0.0,0.3,nH)
    Jpm=-0.03
    hn = h110
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1 = graph_2S_rho(E, Jpm, h, hn, BZres, rank, size, tol)

    if rank == 0:
        f1 = filename
        np.savetxt(f1 + ".txt", d1)
        plt.imshow(d1.T, interpolation="lanczos", origin='lower', extent=[0, 0.3, 0, 2], aspect='auto', cmap='gnuplot')
        plt.ylabel(r'$\omega/J_{yy}$')
        plt.xlabel(r'$h/J_{yy}$')
        plt.savefig(filename)
        plt.clf()


def TwoSpinonDOS_111_a(nH, BZres, filename):
    if not MPI.Is_initialized():
        MPI.Init()
    start = 0.25
    end = 0.9
    E = np.linspace(start, end, 200)
    tol = (end-start)/200
    h = np.linspace(0,0.4,nH)
    Jpm=0.03
    hn = h111
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1 = graph_2S_rho(E, Jpm, h, hn, BZres, rank, size, tol)

    if rank == 0:
        f1 = filename
        np.savetxt(f1 + ".txt", d1)
        plt.imshow(d1.T, interpolation="lanczos", origin='lower', extent=[0, 0.4, start*2, end*2], aspect='auto', cmap='gnuplot')
        plt.ylabel(r'$\omega/J_{yy}$')
        plt.xlabel(r'$h/J_{yy}$')
        plt.savefig(filename)
        plt.clf()
        # DSSFgraph(d1.T, f1, X, Y)
def TwoSpinonDOS_001_a(nH, BZres, filename):
    if not MPI.Is_initialized():
        MPI.Init()
    E = np.linspace(0, 1.5, 200)
    tol = 1.5/200
    h = np.linspace(0,0.1,nH)
    Jpm=-0.3
    hn = h001
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1 = graph_2S_rho(E, Jpm, h, hn, BZres, rank, size, tol)

    if rank == 0:
        f1 = filename
        np.savetxt(f1 + ".txt", d1)
        plt.imshow(d1.T, interpolation="lanczos", origin='lower', extent=[0, 0.1, 0, 3], aspect='auto', cmap='gnuplot')
        plt.ylabel(r'$\omega/J_{yy}$')
        plt.xlabel(r'$h/J_{yy}$')
        plt.savefig(filename)
        plt.clf()
def TwoSpinonDOS_110_a(nH, BZres, filename):
    if not MPI.Is_initialized():
        MPI.Init()
    E = np.linspace(0, 1.5, 200)
    tol = 1/200
    h = np.linspace(0.0,0.23,nH)
    Jpm=-0.3
    hn = h110
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1 = graph_2S_rho(E, Jpm, h, hn, BZres, rank, size, tol)

    if rank == 0:
        f1 = filename
        np.savetxt(f1 + ".txt", d1)
        plt.imshow(d1.T, interpolation="lanczos", origin='lower', extent=[0, 0.23, 0, 3], aspect='auto', cmap='gnuplot')
        plt.ylabel(r'$\omega/J_{yy}$')
        plt.xlabel(r'$h/J_{yy}$')
        plt.savefig(filename)
        plt.clf()

def TwoSpinonDOS_111_b(nH, BZres, filename):
    if not MPI.Is_initialized():
        MPI.Init()
    E = np.linspace(0, 1, 200)
    tol = 1/200
    h = np.linspace(0,0.4,nH)
    Jpm=0.03
    hn = h111
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1 = graph_2S_rho(E, Jpm, h, hn, BZres, rank, size, tol)

    if rank == 0:
        f1 = filename
        np.savetxt(f1 + ".txt", d1)
        plt.imshow(d1.T, interpolation="lanczos", origin='lower', extent=[0, 0.4, 0, 2], aspect='auto', cmap='gnuplot')
        plt.ylabel(r'$\omega/J_{yy}$')
        plt.xlabel(r'$h/J_{yy}$')
        plt.savefig(filename)
        plt.clf()
        # DSSFgraph(d1.T, f1, X, Y)
def pedantic_DSSF_graph_helper(graphMethod, d1, f1, Hr, Lr, dir, lowedge, upedge, dmax):
    for i in range(4):
        for j in range(4):
            tempF = f1+str(i)+str(j)
            np.savetxt(tempF + '.txt', d1[:,:,i,j])
            graphMethod(d1[:,:,i,j]/dmax, tempF, Hr, Lr, lowedge, upedge)
    if (dir==h110).all():
        gp = d1[:,:,0,0] + d1[:,:,0,3] + d1[:,:,3,0] + d1[:,:,3,3]
        gup = d1[:,:,1,1] + d1[:,:,1,2] + d1[:,:,2,1] + d1[:,:,2,2]
        gcorre = d1[:,:,0,1] + d1[:,:,0,2] + d1[:,:,1,0] + d1[:,:,2,0] + d1[:,:,3,1] + d1[:,:,3,2] + d1[:,:,1,3] + d1[:,:,2,3]
        np.savetxt(f1+"polarized.txt", gp)
        np.savetxt(f1+"unpolarized.txt", gup)
        np.savetxt(f1+"polar_unpolar.txt", gcorre)
        graphMethod(gp/dmax, f1+"polarized", Hr, Lr, lowedge, upedge)
        graphMethod(gup/dmax, f1+"unpolarized", Hr, Lr, lowedge, upedge)
        graphMethod(gcorre/dmax, f1+"polar_unpolar", Hr, Lr, lowedge, upedge)
    elif (dir==h111).all():
        gKagome = d1[:,:,1,1] + d1[:,:,1,2] + d1[:,:,1,3] + d1[:,:,2,1] + d1[:,:,3, 1] + d1[:,:,2,2] + d1[:,:,2,3] + d1[:,:,3,2] + d1[:,:,3,3]
        gTri = d1[:,:,0,0]
        gKagomeTri = d1[:,:,0,1] + d1[:,:,0,2] + d1[:,:,0,3] + d1[:,:,1,0] + d1[:,:,2,0] + d1[:,:,3,0]
        np.savetxt(f1+"Kagome.txt", gKagome)
        np.savetxt(f1+"Triangular.txt", gTri)
        np.savetxt(f1+"Kagome-Tri.txt", gKagomeTri)
        graphMethod(gKagome/dmax, f1+"Kagome", Hr, Lr, lowedge, upedge)
        graphMethod(gTri/dmax, f1+"Triangular", Hr, Lr, lowedge, upedge)
        graphMethod(gKagomeTri/dmax, f1+"Kagome-Tri", Hr, Lr, lowedge, upedge)
    else:
        gp = d1[:,:,0,0] + d1[:,:,0,3] + d1[:,:,3,0] + d1[:,:,3,3]
        gup = d1[:,:,1,1] + d1[:,:,1,2] + d1[:,:,2,1] + d1[:,:,2,2]
        gcorre = d1[:,:,0,1] + d1[:,:,0,2] + d1[:,:,1,0] + d1[:,:,2,0] + d1[:,:,3,1] + d1[:,:,3,2] + d1[:,:,1,3] + d1[:,:,2,3]
        np.savetxt(f1+"polarized.txt", gp)
        np.savetxt(f1+"unpolarized.txt", gup)
        np.savetxt(f1+"polar_unpolar.txt", gcorre)
        graphMethod(gp/dmax, f1+"polarized", Hr, Lr, lowedge, upedge)
        graphMethod(gup/dmax, f1+"unpolarized", Hr, Lr, lowedge, upedge)
        graphMethod(gcorre/dmax, f1+"polar_unpolar", Hr, Lr, lowedge, upedge)

def DSSF_pedantic(nE, Jxx, Jyy, Jzz, h, n, flux, BZres, filename,theta=0,beta=0):
    pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux,theta=theta)
    py0s.solvemeanfield()
    kk = np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW1, W1X1, X1Gamma))
    lowedge, upedge = py0s.loweredge(), py0s.upperedge()
    emin, emax = np.min(lowedge), np.max(upedge)
    e = np.linspace(max(emin *0.95, 0), emax *1.02, nE)
    tol = (1.02*emax-max(emin *0.95, 0))/nE*4
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1, d2, d3, d4 = graph_DSSF_pedantic(py0s, e, kk, tol, rank, size, bete)
    # d1= graph_DSSF_pedantic(py0s, e, kk, tol, rank, size)

    if rank == 0:
        pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
        f1 = filename + "/Szz/"
        f2 = filename + "/Szzglobal/"
        f3 = filename + "/Sxx/"
        f4 = filename + "/Sxxglobal/"
        pathlib.Path(f1).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f2).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f3).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f4).mkdir(parents=True, exist_ok=True)
        kline = np.concatenate((graphGammaX, graphXW, graphWK, graphKGamma, graphGammaL, graphLU, graphUW1, graphW1X1, graphX1Gamma))


        Szz = contract('iwjk->iw', d1)
        Szzglobal = contract('iwjk->iw', d2)
        Sxx = contract('iwjk->iw', d3)
        Sxxglobal = contract('iwjk->iw', d4)

        np.savetxt(filename+"/Szz.txt", Szz)
        np.savetxt(filename+"/Szzglobal.txt", Szzglobal)
        np.savetxt(filename+"/Sxx.txt", Sxx)
        np.savetxt(filename+"/Sxxglobal.txt", Sxxglobal)

        DSSFgraph_pedantic(np.abs(Szz/np.max(Szz)), filename+"/Szz", kline, e, lowedge, upedge)
        DSSFgraph_pedantic(np.abs(Szzglobal/np.max(Szzglobal)), filename+"/Szzglobal", kline, e, lowedge, upedge)
        DSSFgraph_pedantic(np.abs(Sxx/np.max(Sxx)), filename+"/Sxx", kline, e, lowedge, upedge)
        DSSFgraph_pedantic(np.abs(Sxxglobal/np.max(Sxxglobal)), filename+"/Sxxglobal", kline, e, lowedge, upedge)


        pedantic_DSSF_graph_helper(DSSFgraph_pedantic, d1, f1, kline, e, n, lowedge, upedge, np.max(Szz))
        pedantic_DSSF_graph_helper(DSSFgraph_pedantic, d2, f2, kline, e, n, lowedge, upedge, np.max(Szzglobal))
        pedantic_DSSF_graph_helper(DSSFgraph_pedantic, d3, f3, kline, e, n, lowedge, upedge, np.max(Sxx))
        pedantic_DSSF_graph_helper(DSSFgraph_pedantic, d3, f4, kline, e, n, lowedge, upedge, np.max(Sxxglobal))

def samplegraph(nK, filenames):
    fig, axs = plt.subplots(3, len(filenames))

    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    for i in range(len(filenames)):
        f1 = "Files/" + filenames[i] + "_local"
        f2 = "Files/" + filenames[i] + "_global"
        f3 = "Files/" + filenames[i] + "_NSF"
        d1 = np.loadtxt(f1 + '.txt')
        d2 = np.loadtxt(f2 + '.txt')
        d3 = np.loadtxt(f3 + '.txt')
        d = [d1 / np.max(d1), d2 / np.max(d2), d3 / np.max(d3)]
        for j in range(3):
            axs[j, i].pcolormesh(A, B, d[j])
            axs[j, i].set_ylabel(r'$(0,0,L)$')
            axs[j, i].set_xlabel(r'$(H,H,0)$')
    plt.show()

def SSSF_line(nK, Jxx, Jyy, Jzz, hmin, hmax, nH, n, flux, BZres, dirname):
    hs = np.linspace(hmin, hmax, nH)
    dirString = ""
    scatplane = ""
    if (n==np.array([0,0,1])).all():
        dirString = "001"
        scatplane="hk0"
    elif (n==np.array([1,1,0])/np.sqrt(2)).all():
        dirString = "110"
        scatplane="hhl"
    else:
        dirString = "111"
        scatplane="hhk"
    for i in range(nH):
        filename = dirname+"/h_" + dirString + "/h=" + str(hs[i]) + "/"
        pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
        SSSF(nK, Jxx, Jyy, Jzz, hs[i], n, flux, BZres, filename, scatplane)

def SSSF_line_pedantic(nK, Jxx, Jyy, Jzz, hmin, hmax, nH, n, flux, BZres, dirname, scatplane, K=0, Hr=2.5, Lr=2.5):
    hs = np.linspace(hmin, hmax, nH)
    dirString = ""
    if (n==np.array([0,0,1])).all():
        dirString = "001"
    elif (n==np.array([1,1,0])/np.sqrt(2)).all():
        dirString = "110"
    else:
        dirString = "111"
    for i in range(nH):
        filename = dirname+"/h_" + dirString + "/h=" + str(hs[i]) + "/"
        pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
        SSSF_pedantic(nK, Jxx, Jyy, Jzz, hs[i], n, flux, BZres, filename, scatplane, K=K, Hr=Hr, Lr=Lr)

def DSSF_line(nE, Jxx, Jyy, Jzz, hmin, hmax, nH, n, flux, BZres, dirname):
    hs = np.linspace(hmin, hmax, nH)
    dirString = ""
    if (n==np.array([0,0,1])).all():
        dirString = "001"
    elif (n==np.array([1,1,0])/np.sqrt(2)).all():
        dirString = "110"
    else:
        dirString = "111"
    for i in range(nH):
        filename = dirname+"/h_" + dirString + "/h=" + str(hs[i]) + "/"
        pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
        DSSF(nE, Jxx, Jyy, Jzz, hs[i], n, flux, BZres, filename)
def DSSF_line_pedantic(nE, Jxx, Jyy, Jzz, hmin, hmax, nH, n, flux, BZres, dirname):
    hs = np.linspace(hmin, hmax, nH)
    dirString = ""
    if (n==np.array([0,0,1])).all():
        dirString = "001"
    elif (n==np.array([1,1,0])/np.sqrt(2)).all():
        dirString = "110"
    else:
        dirString = "111"
    for i in range(nH):
        filename = dirname+"/h_" + dirString + "/h=" + str(hs[i]) + "/"
        pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
        DSSF_pedantic(nE, Jxx, Jyy, Jzz, hs[i], n, flux, BZres, filename)

# endregion

# region two spinon continuum
def TWOSPINCON(nK, h, n, Jxx, Jyy, Jzz, flux, BZres, filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # if Jpm >= 0:
    #     py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, h=h, n=n, kappa=1)
    # else:

    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, kappa=2, flux=flux)
    py0s.solvemeanfield()

    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hnhltoK(A, B).reshape(3, -1).T

    n = len(K) / size
    left = int(rank * n)
    right = int((rank + 1) * n)

    currK = K[left:right, :]

    sendbuf1 = py0s.minMaxCal(currK)

    recvbuf1 = None

    if rank == 0:
        recvbuf1 = np.zeros((nK * nK, 2))

    sendcounts = np.array(comm.gather(sendbuf1.shape[0] * sendbuf1.shape[1], 0))

    comm.Gatherv(sendbuf=sendbuf1, recvbuf=(recvbuf1, sendcounts), root=0)

    if rank == 0:
        f1 = "Files/" + filename + "_lower"
        f2 = "Files/" + filename + "_upper"
        loweredge = recvbuf1[:, 0]
        upperedge = recvbuf1[:, 1]
        loweredge = loweredge.reshape((nK, nK))
        upperedge = upperedge.reshape((nK, nK))
        np.savetxt(f1 + '.txt', loweredge)
        np.savetxt(f2 + '.txt', upperedge)
        # d1 = np.loadtxt(f1+'.txt')
        # d2 = np.loadtxt(f2 + '.txt')
        TWOSPINONGRAPH(A, B, loweredge, f1)
        TWOSPINONGRAPH(A, B, upperedge, f2)



def TWOSPINCON_general(nK, h, n, Jxx, Jyy, Jzz, BZres, flux, filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # if Jpm >= 0:
    #     py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, h=h, n=n, kappa=1)
    # else:

    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, kappa=2, flux=flux)
    py0s.solvemeanfield()

    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hnhltoK(A, B).reshape(3, -1).T

    n = len(K) / size
    left = int(rank * n)
    right = int((rank + 1) * n)

    currK = K[left:right, :]

    sendbuf1 = py0s.minMaxCal(currK)

    recvbuf1 = None

    if rank == 0:
        recvbuf1 = np.zeros((nK * nK, 2))

    sendcounts = np.array(comm.gather(sendbuf1.shape[0] * sendbuf1.shape[1], 0))

    comm.Gatherv(sendbuf=sendbuf1, recvbuf=(recvbuf1, sendcounts), root=0)

    if rank == 0:
        f1 = "Files/" + filename + "_lower"
        f2 = "Files/" + filename + "_upper"
        loweredge = recvbuf1[:, 0]
        upperedge = recvbuf1[:, 1]
        loweredge = loweredge.reshape((nK, nK))
        upperedge = upperedge.reshape((nK, nK))
        np.savetxt(f1 + '.txt', loweredge)
        np.savetxt(f2 + '.txt', upperedge)
        # d1 = np.loadtxt(f1+'.txt')
        # d2 = np.loadtxt(f2 + '.txt')
        TWOSPINONGRAPH(A, B, loweredge, f1)
        TWOSPINONGRAPH(A, B, upperedge, f2)



def TWOSPINONGRAPH(A, B, d1, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.plot_surface(A, B, d1, cmap=cm.coolwarm)
    plt.savefig(filename + ".png")
    plt.clf()
    plt.pcolormesh(A, B, d1)
    plt.savefig(filename + "_flatmesh.png")
    plt.clf()
# endregion

# endregion









