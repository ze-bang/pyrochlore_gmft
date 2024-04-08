# import pyrochlore_dispersion_pi_gang_chen as pygang
from archive import pyrochlore_dispersion_pi_old as pypipyold
from misc_helper import *
import matplotlib.pyplot as plt
from matplotlib import cm
import pyrochlore_gmft as pycon
from mpi4py import MPI
import os
import pathlib

def delta(Ek, Eq, omega, tol):
    beta = 0
    size = Ek.shape[1]
    Ekenlarged = contract('ik,j->ikj', Ek, np.ones(size))
    Eqenlarged = contract('ik,j->ijk', Eq, np.ones(size))
    A = contract('ia, ib, iab->iab', 1 + bose(beta, Ek), 1 + bose(beta, Eq),
                 cauchy(omega - Ekenlarged - Eqenlarged, tol))
    B = contract('ia, ib, iab->iab', 1 + bose(beta, Ek), bose(beta, Eq), cauchy(omega - Ekenlarged + Eqenlarged, tol))
    C = contract('ia, ib, iab->iab', bose(beta, Ek), 1 + bose(beta, Eq), cauchy(omega + Ekenlarged - Eqenlarged, tol))
    D = contract('ia, ib, iab->iab', bose(beta, Ek), bose(beta, Eq), cauchy(omega + Ekenlarged + Eqenlarged, tol))
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

def Spm_Spp_omega(Ks, Qs, q, omega, tol, pyp0, lam=0):
    # greenpK, tempE = pyp0.green_pi_branch(Ks, lam)
    # greenpQ, tempQ = pyp0.green_pi_branch(Qs, lam)
    greenpK, tempE, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here, unitcell = pyp0.green_pi_branch_reduced(Ks)
    greenpQ, tempQ, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here, unitcell = pyp0.green_pi_branch_reduced(Qs)

    if pyp0.Jpmpm ==0:
        size = int(tempE.shape[1]/2)
    else:
        size = int(tempE.shape[1]/4)

    Kreal = contract('ij,jk->ik',Ks-q/2, BasisBZA)
    deltapm = deltas(tempE, tempQ, omega, tol)

    ffact = contract('ik, jlk->ijl', Kreal, NNminus)
    ffactpm = np.exp(1j * ffact)
    ffact = contract('ik, jlk->ijl', Kreal, NNplus)
    ffactpp = np.exp(1j * ffact)
    Spm = contract('ioab, ipyx, iwop, abjk, jax, kby, ijk->wijk', greenpK[:, :, 0:size, 0:size], greenpQ[:, :, size:2*size, size:2*size],
                   deltapm, A_pi_rs_rsp_here, unitcell, unitcell,
                   ffactpm) / 64

    Spp = contract('ioay, ipbx, iwop, abjk, jax, kby, ijk->wijk', greenpK[:, :, 0:size, size:2*size], greenpQ[:, :, 0:size, size:2*size],
                   deltapm, A_pi_rs_rsp_pp_here, unitcell, unitcell,
                   ffactpp) / 64
    if not pyp0.Jpmpm == 0:
        deltapairing = deltas_pairing(tempE, omega, tol)
        Spp = Spp + contract('ioab, ipyx, iwo, abjk, jax, kby, ijk->ijk', greenpK[:, :, 0:size, 2*size:3*size], greenpQ[:, :, 3*size:4*size, size:2*size],
                             deltapairing, pyp0.A_pi_rs_rsp_pp_here, unitcell, unitcell, ffactpm) / 64
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
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)

    return rectemp, rectemp1, rectemp2, rectemp3


# endregion

# region SSSF
def SpmSpp(K, Q, q, pyp0, lam=0):
    greenpK = pyp0.green_pi(K, lam)
    greenpQ = pyp0.green_pi(Q, lam)
    Kreal = contract('ij,jk->ik',K-q/2, BasisBZA)

    if pyp0.Jpmpm ==0:
        size = int(greenpK.shape[1]/2)
    else:
        size = int(greenpK.shape[1]/4)

    ffactpm = np.exp(1j * contract('ik, jlk->ijl', Kreal, NNminus))
    ffactpp = np.exp(1j * contract('ik, jlk->ijl', Kreal, NNplus))

    Spm = contract('iab, iyx, abjk, jax, kby, ijk->ijk', greenpK[:, 0:size, 0:size], greenpQ[:, size:2*size, size:2*size], pyp0.A_pi_rs_rsp_here,
                   piunitcell, piunitcell,
                   ffactpm) / 64

    Spp = contract('iay, ibx, abjk, jax, kby, ijk->ijk', greenpK[:, 0:size, size:2*size], greenpQ[:, 0:size, size:2*size], pyp0.A_pi_rs_rsp_pp_here,
                   piunitcell, piunitcell,
                   ffactpp) / 64

    if not pyp0.Jpmpm == 0:
        Spp = Spp + contract('iab, iyx, abjk, jax, kby, ijk->ijk', greenpK[:, 0:size, 2*size:3*size], greenpQ[:, 3*size:4*size, size:2*size], pyp0.A_pi_rs_rsp_pp_here,
                   piunitcell, piunitcell,
                   ffactpm) / 64
    return Spm, Spp
def SSSF_core(q, v, pyp0):
    Ks = pyp0.pts
    Qs = Ks - q

    if (v==0).all():
        v = np.array([-1,1,0])/np.sqrt(2)
    else:
        v = v/np.linalg.norm(v)

    Spm, Spp = SpmSpp(Ks, Qs, q, pyp0, lam=pyp0.lams)
    Szz = (np.real(Spm) + np.real(Spp)) / 2
    Sxx = (np.real(Spm) - np.real(Spp)) / 2

    qreal = contract('j,jk->k',q, BasisBZA)
    Sglobalzz = contract('ijk,jk,i->', Szz, g(qreal), pyp0.weights)
    SNSFzz = contract('ijk,jk,i->', Szz, gNSF(v), pyp0.weights)
    Szz = contract('ijk,i->', Szz, pyp0.weights)
    Sglobalxx = contract('ijk,jk,i->', Sxx, g(qreal), pyp0.weights)
    SNSFxx = contract('ijk,jk,i->', Sxx, gNSF(v), pyp0.weights)
    Sxx = contract('ijk,i->', Sxx, pyp0.weights)

    return Szz, Sglobalzz, SNSFzz, Sxx, Sglobalxx, SNSFxx
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

    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None
    rectemp5 = None

    if rank == 0:
        rectemp = np.zeros(len(K), dtype=np.float64)
        rectemp1 = np.zeros(len(K), dtype=np.float64)
        rectemp2 = np.zeros(len(K), dtype=np.float64)
        rectemp3 = np.zeros(len(K), dtype=np.float64)
        rectemp4 = np.zeros(len(K), dtype=np.float64)
        rectemp5 = np.zeros(len(K), dtype=np.float64)

    for i in range(currsizeK):
        sendtemp[i], sendtemp1[i], sendtemp2[i], sendtemp3[i], sendtemp4[i], sendtemp5[i] = SSSF_core(currK[i], V[i], pyp0)

    sendcounts = np.array(comm.gather(len(sendtemp), 0))
    sendcounts1 = np.array(comm.gather(len(sendtemp1), 0))
    sendcounts2 = np.array(comm.gather(len(sendtemp2), 0))
    sendcounts3 = np.array(comm.gather(len(sendtemp3), 0))
    sendcounts4 = np.array(comm.gather(len(sendtemp4), 0))
    sendcounts5 = np.array(comm.gather(len(sendtemp5), 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)

    return rectemp, rectemp1, rectemp2, rectemp3, rectemp4, rectemp5


# endregion

# region Graphing
def DSSFgraph(A, B, D, py0s, filename):
    plt.imshow(D, interpolation="lanczos",extent =[A.min(), A.max(), B.min(), B.max()])
    plt.ylabel(r'$\omega/J_{zz}$')
    py0s.graph_loweredge(False)
    py0s.graph_upperedge(False)
    plt.savefig(filename + ".pdf")
    plt.clf()

def plot_line(A, B, color):
    temp = np.array([A,B]).T
    plt.plot(temp[0], temp[1], color,zorder=0)

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

# Gamma = np.array([0, 0, 0])
# L = np.array([1, 1, 1])/2
# X = np.array([0, 0.5, 0.5])
# W = np.array([0.25, 0.75, 0.5])
# K = np.array([0.375, 0.75, 0.375])
# U = np.array([0.25, 0.625, 0.625])

# print(0.5*BasisBZA[0]+0.5*BasisBZA[1]+1*BasisBZA[2])
# print(-0.5*BasisBZA[0]+0.5*BasisBZA[1]+0*BasisBZA[2])
# print(1*np.array([0.5,0.5,1])-1*np.array([-0.5,0.5,0]))
def SSSFGraphHKK(A, B, d1, filename):
    plt.pcolormesh(A, B, d1)
    plt.colorbar()
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
    plt.scatter(Gamms[0,0],Gamms[0,1])
    plt.scatter(Xs[:,0], Xs[:,1])
    plt.scatter(Ks[:,0], Ks[:,1])
    plt.scatter(Ws[:, 0], Ws[:, 1])

    plot_BZ_hkk(Gamms[0], Boundary, 'b:')


    plt.ylabel(r'$(K,-K,0)$')
    plt.xlabel(r'$(H,H,0)$')
    plt.xlim([-2.5,2.5])
    plt.ylim([-2.5,2.5])
    # plt.show()
    plt.savefig(filename + ".pdf")
    plt.clf()
def SSSFGraphHHL(A, B, d1, filename):
    plt.pcolormesh(A, B, d1)
    plt.colorbar()
    Gamms = np.array([[0,0],[1,1],[-1,1],[1,-1],[-1,-1],[2,0],[0,2],[-2,0],[0,-2],[2,2],[-2,2],[2,-2],[-2,-2]])
    Ls = np.array([[0.5,0.5]])
    Xs = np.array([[0,1]])
    Us = np.array([[0.25,1]])
    Ks = np.array([[0.75,0]])


    Boundary = np.array([[0.25, 1],[-0.25,1],[-0.75,0],[-0.25,-1],[0.25,-1],[0.75,0]])



    plot_BZ_hhl(Gamms[0], Boundary, 'b:')

    plt.scatter(Gamms[0,0], Gamms[0,1],zorder=1)
    plt.scatter(Ls[:,0], Ls[:,1],zorder=1)
    plt.scatter(Xs[:, 0], Xs[:, 1],zorder=1)
    plt.scatter(Ks[:, 0], Ks[:, 1],zorder=1)
    plt.scatter(Us[:, 0], Us[:, 1],zorder=1)
    plot_text(Gamms,r'$\Gamma$')
    plot_text(Ls,r'$L$')
    plot_text(Xs,r'$X$')
    plot_text(Us,r'$U$')
    plot_text(Ks,r'$K$')

    plt.ylabel(r'$(0,0,L)$')
    plt.xlabel(r'$(H,H,0)$')
    plt.xlim([-2.5,2.5])
    plt.ylim([-2.5,2.5])
    # plt.show()
    plt.savefig(filename + ".pdf")
    plt.clf()
def SSSFGraphHK0(A, B, d1, filename):
    plt.pcolormesh(A, B, d1)
    plt.colorbar()

    Gamms = np.array([[0,0],[2,0],[0,2],[-2,0],[0,-2],[2,2],[-2,2],[2,-2],[-2,-2]])
    Xs = np.array([[1, 0]])
    Ks = np.array([[1,1]])*0.375/0.5
    Ws = np.array([[1,0.5]])

    Boundary = np.array([[1, 0.5], [0.5,1], [-0.5,1], [-1,0.5],[-1,-0.5],[-0.5,-1],[0.5,-1],[1,-0.5]])

    plt.scatter(Gamms[0,0], Gamms[0,1],zorder=1)
    plt.scatter(Xs[:, 0], Xs[:, 1], zorder=1)
    plt.scatter(Ks[:, 0], Ks[:, 1], zorder=1)
    plt.scatter(Ws[:, 0], Ws[:, 1], zorder=1)
    plot_text(Gamms,r'$\Gamma$')
    plot_text(Xs,r'$X$')
    plot_text(Ks,r'$K$')
    plot_text(Ws,r'$W$')

    plot_BZ_hkk(Gamms[0], Boundary, 'b:')

    plt.ylabel(r'$(0,K,0)$')
    plt.xlabel(r'$(H,0,0)$')
    plt.xlim([-2.5,2.5])
    plt.ylim([-2.5,2.5])
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

    d1, d2, d3, d4, d5, d6 = graph_SSSF(py0s, K, v, rank, size)
    if rank == 0:
        f1 = filename + "Szz_local"
        f2 = filename + "Szz_global"
        f3 = filename + "Szz_NSF"
        f4 = filename + "Sxx_local"
        f5 = filename + "Sxx_global"
        f6 = filename + "Sxx_NSF"
        np.savetxt(f1 + '.txt', d1)
        np.savetxt(f2 + '.txt', d2)
        np.savetxt(f3 + '.txt', d3)
        np.savetxt(f4 + '.txt', d4)
        np.savetxt(f5 + '.txt', d5)
        np.savetxt(f6 + '.txt', d6)

def SSSF(nK, Jxx, Jyy, Jzz, h, n, flux, BZres, filename, hkl, K=0):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)

    if hkl == "hk0":
        K = hkztoK(A, B, K).reshape((nK*nK,3))
        scatterPlane = hk0scaplane(A, B).reshape((nK*nK,3))
    elif hkl=="hhl":
        K = hhltoK(A, B, K).reshape((nK*nK,3))
        scatterPlane = hhlscaplane(A, B).reshape((nK*nK,3))
    else:
        K = hkktoK(A, B, K).reshape((nK*nK,3))
        scatterPlane = hkkscaplane(A, B).reshape((nK*nK,3))

    v = np.zeros(scatterPlane.shape)
    v[:,0] = -scatterPlane[:,1]
    v[:,1] = scatterPlane[:,0]

    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d1, d2, d3, d4, d5, d6 = graph_SSSF(py0s, K, v, rank, size)
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
        if hkl==0:
            SSSFGraphHK0(A, B, d1, f1)
            SSSFGraphHK0(A, B, d2, f2)
            SSSFGraphHK0(A, B, d3, f3)
            SSSFGraphHK0(A, B, d4, f4)
            SSSFGraphHK0(A, B, d5, f5)
            SSSFGraphHK0(A, B, d6, f6)
        elif hkl==1:
            SSSFGraphHHL(A, B, d1, f1)
            SSSFGraphHHL(A, B, d2, f2)
            SSSFGraphHHL(A, B, d3, f3)
            SSSFGraphHHL(A, B, d4, f4)
            SSSFGraphHHL(A, B, d5, f5)
            SSSFGraphHHL(A, B, d6, f6)
        else:
            SSSFGraphHKK(A, B, d1, f1)
            SSSFGraphHKK(A, B, d2, f2)
            SSSFGraphHKK(A, B, d3, f3)
            SSSFGraphHKK(A, B, d4, f4)
            SSSFGraphHKK(A, B, d5, f5)
            SSSFGraphHKK(A, B, d6, f6)


def SSSF_HHL_KK_integrated(nK, Jxx, Jyy, Jzz, h, n, flux, Lmin, Lmax, Ln, BZres, filename):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
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
        SSSFGraphHHL(A, B, d1, f1)
        SSSFGraphHHL(A, B, d2, f2)
        SSSFGraphHHL(A, B, d3, f3)
        SSSFGraphHHL(A, B, d4, f4)
        SSSFGraphHHL(A, B, d5, f5)
        SSSFGraphHHL(A, B, d6, f6)

def SSSF_HHKnK_L_integrated(nK, Jxx, Jyy, Jzz, h, n, flux, Lmin, Lmax, Ln, BZres, filename):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    H = np.linspace(-2.5, 2.5, nK)
    K = np.linspace(-2.5, 2.5, nK)
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
        SSSFGraphHKK(A, B, d1, f1)
        SSSFGraphHKK(A, B, d2, f2)
        SSSFGraphHKK(A, B, d3, f3)
        SSSFGraphHKK(A, B, d4, f4)
        SSSFGraphHKK(A, B, d5, f5)
        SSSFGraphHKK(A, B, d6, f6)

def SSSF_HK0_L_integrated(nK, Jxx, Jyy, Jzz, h, n, flux, Lmin, Lmax, Ln, BZres, filename):

    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    H = np.linspace(-2.5, 2.5, nK)
    K = np.linspace(-2.5, 2.5, nK)
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
        SSSFGraphHK0(A, B, d1, f1)
        SSSFGraphHK0(A, B, d2, f2)
        SSSFGraphHK0(A, B, d3, f3)
        SSSFGraphHK0(A, B, d4, f4)
        SSSFGraphHK0(A, B, d5, f5)
        SSSFGraphHK0(A, B, d6, f6)
def DSSF(nE, Jxx, Jyy, Jzz, h, n, flux, BZres, filename):
    py0s = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)
    py0s.solvemeanfield()
    kk = np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW))
    emin, emax = py0s.TWOSPINON_DOMAIN(kk)
    e = np.arange(max(emin *0.95, 0), emax *1.02, nE)
    tol = nE / 2
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
        kline = np.concatenate((graphGammaX, graphXW, graphWK, graphKGamma, graphGammaL, graphLU, graphUW))
        X, Y = np.meshgrid(kline, e)
        DSSFgraph(X, Y, d1.T, py0s, f1)
        DSSFgraph(X, Y, d2.T, py0s, f2)
        DSSFgraph(X, Y, d3.T, py0s, f3)
        DSSFgraph(X, Y, d4.T, py0s, f4)

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
    K = hhltoK(A, B).reshape(3, -1).T

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
    K = hhltoK(A, B).reshape(3, -1).T

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









