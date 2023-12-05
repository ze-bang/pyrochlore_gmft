
from pyrochlore_dispersion_pi import green_pi, green_pi_branch
from pyrochlore_dispersion import green_zero_branch, green_zero
import pyrochlore_dispersion_pi_gang_chen as pygang
import pyrochlore_dispersion_pi_old as pypipyold
import pyrochlore_general as pygen
from misc_helper import *
import matplotlib.pyplot as plt
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
from matplotlib import cm


def delta(Ek, Eq, omega, tol):
    beta = 0
    size = Ek.shape[1]
    Ekenlarged = contract('ik,j->ikj',Ek, np.ones(size))
    Eqenlarged = contract('ik,j->ijk', Eq, np.ones(size))
    A = contract('ia, ib, iab->iab' ,1+bose(beta, Ek) ,1+bose(beta, Eq), cauchy(omega - Ekenlarged - Eqenlarged, tol))
    B = contract('ia, ib, iab->iab' ,1+bose(beta, Ek) ,bose(beta, Eq), cauchy(omega - Ekenlarged + Eqenlarged, tol))
    C = contract('ia, ib, iab->iab' ,bose(beta, Ek) ,1+bose(beta, Eq), cauchy(omega + Ekenlarged - Eqenlarged, tol))
    D = contract('ia, ib, iab->iab', bose(beta, Ek), bose(beta, Eq), cauchy(omega + Ekenlarged + Eqenlarged, tol))
    return A+B+C+D



#region DSSF


def Spm_zero_DSSF(K, Q, q, omega, tol, pyp0, lam = 0):

    greenp1, tempE = pyp0.green_zero_branch(K, lam)
    greenp2, tempQ = pyp0.green_zero_branch(Q, lam)

    #region S+- and S-+
    deltapm = delta(tempE, tempQ, omega, tol)
    ffact = contract('ik, jlk->ijl', K - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)

    greenA = contract('ia, ib, iab->i',greenp1[:, :, 0,0], greenp2[:,:, 1,1], deltapm)
    greenpm = contract('i,ijk->ijk',greenA, ffactpm)/4
    #endregion

    #region S++ and S--
    ffact = contract('ik, jlk->ijl', K - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)

    greenB = contract('ia, ib, iab->i',greenp1[:, :, 0,1], greenp2[:,:, 1,0], deltapm)
    greenpp = contract('i,ijk->ijk',greenB, ffactpp)/4

    return greenpm, greenpp


def DSSF_zero(q, omega, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks-q

    greenpm, greenpp = Spm_zero_DSSF(Ks, Qs, q, omega, tol, pyp0)

    Szz = (np.real(greenpp) + np.real(greenpm))/2
    Sxx = (-np.real(greenpp) + np.real(greenpm))/2

    Sglobalzz = np.mean(contract('ijk,jk->i', Szz, g(q)))
    Szz = np.mean(contract('ijk->i',Szz))

    Sglobalxx = np.mean(contract('ijk,jk->i', Sxx, g(q)))
    Sxx = np.mean(contract('ijk->i',Sxx))
    return Szz, Sglobalzz, Sxx, Sglobalxx

def Spm_pi_DSSF(Ks, Qs, q, omega, tol, pyp0, lam = 0):

    greenpK, tempE = pyp0.green_pi_branch(Ks, lam)
    greenpQ, tempQ = pyp0.green_pi_branch(Qs, lam)

    deltapm = delta(tempE, tempQ, omega, tol)

    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)

    Spm = contract('ioab, ipyx, iop, abjk, jax, kby, ijk->ijk', greenpK[:, :, 0:4, 0:4], greenpQ[:, :, 4:8, 4:8],
                   deltapm, A_pi_rs_rsp, piunitcell, piunitcell,
                   ffactpm) / 4

    Spp = contract('ioax, ipby, iop, abjk, jax, kby, ijk->ijk', greenpK[:, :, 0:4, 4:8], greenpQ[:, :, 0:4, 4:8],
                   deltapm, A_pi_rs_rsp_pp, piunitcell, piunitcell,
                   ffactpp) / 4


    return Spm, Spp


def DSSF_pi(q, omega, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks - q


    Spm, Spp = Spm_pi_DSSF(Ks, Qs, q, omega, tol, pyp0)

    Szz = (np.real(Spm) + np.real(Spp))/2/4
    Sxx = (np.real(Spm) - np.real(Spp))/2/4

    Sglobalzz = np.mean(contract('ijk,jk->i', Szz, g(q)))
    Szz = np.mean(contract('ijk->i',Szz))

    Sglobalxx = np.mean(contract('ijk,jk->i', Sxx, g(q)))
    Sxx = np.mean(contract('ijk->i',Sxx))
    return Szz, Sglobalzz, Sxx, Sglobalxx


def Spm_pi_DSSF(Ks, Qs, q, omega, tol, pyp0, lam = 0):

    greenpK, tempE = pyp0.green_pi_branch(Ks, lam)
    greenpQ, tempQ = pyp0.green_pi_branch(Qs, lam)

    deltapm = delta(tempE, tempQ, omega, tol)

    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)

    Spm = contract('ioab, ipyx, iop, abjk, jax, kby, ijk->ijk', greenpK[:, :, 0:4, 0:4], greenpQ[:, :, 4:8, 4:8],
                   deltapm, A_pi_rs_rsp, piunitcell, piunitcell,
                   ffactpm) / 4

    Spp = contract('ioax, ipby, iop, abjk, jax, kby, ijk->ijk', greenpK[:, :, 0:4, 4:8], greenpQ[:, :, 0:4, 4:8],
                   deltapm, A_pi_rs_rsp_pp, piunitcell, piunitcell,
                   ffactpp) / 4


    return Spm, Spp


def DSSF_pi(q, omega, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks - q


    Spm, Spp = Spm_pi_DSSF(Ks, Qs, q, omega, tol, pyp0)

    Szz = (np.real(Spm) + np.real(Spp))/2/4
    Sxx = (np.real(Spm) - np.real(Spp))/2/4

    Sglobalzz = np.mean(contract('ijk,jk->i', Szz, g(q)))
    Szz = np.mean(contract('ijk->i',Szz))

    Sglobalxx = np.mean(contract('ijk,jk->i', Sxx, g(q)))
    Sxx = np.mean(contract('ijk->i',Sxx))
    return Szz, Sglobalzz, Sxx, Sglobalxx


def graph_DSSF_zero(pyp0, E, K, tol, rank, size):
    comm = MPI.COMM_WORLD
    n = len(E)/size

    left = int(rank*n)
    right = int((rank+1)*n)

    currsize = right-left
    
    sendtemp = np.zeros((currsize, len(K)), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, len(K)), dtype=np.float64)
    sendtemp2 = np.zeros((currsize, len(K)), dtype=np.float64)
    sendtemp3 = np.zeros((currsize, len(K)), dtype=np.float64)

    currE = E[left:right]

    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None

    if rank == 0:
        rectemp = np.zeros((len(E), len(K)), dtype=np.float64)
        rectemp1 = np.zeros((len(E), len(K)), dtype=np.float64)
        rectemp2 = np.zeros((len(E), len(K)), dtype=np.float64)
        rectemp3 = np.zeros((len(E), len(K)), dtype=np.float64)

    for i in range(currsize):
        for j in range(len(K)):
            sendtemp[i,j], sendtemp1[i,j], sendtemp2[i,j], sendtemp3[i,j] = DSSF_zero(K[j], currE[i], pyp0, tol)

    sendcounts = np.array(comm.gather(sendtemp.shape[0]*sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0]*sendtemp1.shape[1], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0] * sendtemp2.shape[1], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0] * sendtemp3.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)

    return rectemp, rectemp1

def graph_DSSF_pi(pyp0, E, K, tol, rank, size):

    comm = MPI.COMM_WORLD
    n = len(E)/size

    left = int(rank*n)
    right = int((rank+1)*n)
    
    currsize = right-left
    
    sendtemp = np.zeros((currsize, len(K)), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, len(K)), dtype=np.float64)
    sendtemp2 = np.zeros((currsize, len(K)), dtype=np.float64)
    sendtemp3 = np.zeros((currsize, len(K)), dtype=np.float64)

    currE = E[left:right]

    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None

    if rank == 0:
        rectemp = np.zeros((len(E), len(K)), dtype=np.float64)
        rectemp1 = np.zeros((len(E), len(K)), dtype=np.float64)
        rectemp2 = np.zeros((len(E), len(K)), dtype=np.float64)
        rectemp3 = np.zeros((len(E), len(K)), dtype=np.float64)

    for i in range(currsize):
        for j in range(len(K)):
            sendtemp[i,j], sendtemp1[i,j], sendtemp2[i,j], sendtemp3[i,j] = DSSF_pi(K[j], currE[i], pyp0, tol)

    sendcounts = np.array(comm.gather(sendtemp.shape[0]*sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0]*sendtemp1.shape[1], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0] * sendtemp2.shape[1], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0] * sendtemp3.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)

    return rectemp, rectemp1, rectemp2, rectemp3

#endregion



#region SSSF


def Spm_zero(K, Q, q, pyp0, lam = 0):
    greenp1 = pyp0.green_zero(K, lam)
    greenp2 = pyp0.green_zero(Q, lam)

    # region S+- and S-+
    ffact = contract('ik, jlk->ijl', K - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)

    greenA = greenp1[:, 0, 0] * greenp2[:, 1, 1]
    greenpm = contract('i,ijk->ijk', greenA, ffactpm) / 4
    # endregion

    # region S++ and S--
    ffact = contract('ik, jlk->ijl', K - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)

    greenB = greenp1[:, 0, 1] * greenp2[:, 1, 0]
    greenpp = contract('i,ijk->ijk', greenB, ffactpp) / 4

    return greenpm, greenpp

def SSSF_zero(q, v, pyp0):

    Ks = pyp0.bigB
    Qs = Ks-q
    greenpm, greenpp = Spm_zero(Ks, Qs, q, pyp0)

    Szz = (np.real(greenpp) + np.real(greenpm))/2
    Sxx = (np.real(greenpp) + np.real(greenpm)) / 2

    Sglobalzz = np.mean(contract('ijk,jk->i',Szz, g(q)))
    SNSFzz = np.mean(contract('ijk,jk->i',Szz, gNSF(v)))
    Szz = np.mean(contract('ijk->i',Szz))
    Sglobalxx = np.mean(contract('ijk,jk->i',Sxx, g(q)))
    SNSFxx = np.mean(contract('ijk,jk->i',Sxx, gNSF(v)))
    Sxx = np.mean(contract('ijk->i',Sxx))

    return Szz, Sglobalzz, SNSFzz, Sxx, Sglobalxx, SNSFxx


def Spm_pi(K,Q,q, pyp0, lam=0):

    greenpK = pyp0.green_pi(K, lam)
    greenpQ = pyp0.green_pi(Q, lam)

    ffact = contract('ik, jlk->ijl', K - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)

    ffact = contract('ik, jlk->ijl', K - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)

    Spm = contract('iab, iyx, abjk, jax, kby, ijk->ijk', greenpK[:, 0:4, 0:4], greenpQ[:, 4:8, 4:8], A_pi_rs_rsp,
                   piunitcell, piunitcell,
                   ffactpm) / 4

    Spp = contract('iay, ibx, abjk, jax, kby, ijk->ijk', greenpK[:, 0:4, 4:8], greenpQ[:, 0:4, 4:8], A_pi_rs_rsp_pp,
                   piunitcell, piunitcell,
                   ffactpp) / 4
    return Spm, Spp

def SSSF_pi(q, v, pyp0):
    Ks = pyp0.bigB
    Qs = Ks - q
    v = v / magnitude(v)

    Spm, Spp = Spm_pi(Ks, Qs, q, pyp0, lam=pyp0.lams)
    Szz = (np.real(Spm) + np.real(Spp)) / 2 / 4
    Sxx = (np.real(Spm) - np.real(Spp)) / 2 / 4

    Sglobalzz = np.mean(contract('ijk,jk->i',Szz, g(q)))
    SNSFzz = np.mean(contract('ijk,jk->i',Szz, gNSF(v)))
    Szz = np.mean(contract('ijk->i',Szz))
    Sglobalxx = np.mean(contract('ijk,jk->i',Sxx, g(q)))
    SNSFxx = np.mean(contract('ijk,jk->i',Sxx, gNSF(v)))
    Sxx = np.mean(contract('ijk->i',Sxx))


    return Szz, Sglobalzz, SNSFzz, Sxx, Sglobalxx, SNSFxx


def graph_SSSF_zero(pyp0, K, V, rank, size):

    comm = MPI.COMM_WORLD
    nK = len(K)

    sizeK, sizeH = factors(size, nK)

    nb = nK / sizeK
    nh = nK / sizeH

    rankK = rank % sizeK
    rankH = rank // sizeK

    leftK = int(rankK * nb)
    rightK = int((rankK + 1) * nb)
    currsizeK = rightK - leftK

    leftH = int(rankH * nh)
    rightH = int((rankH + 1) * nh)
    currsizeH = rightH - leftH

    currK = K[leftK:rightK, leftH:rightH, :]

    sendtemp = np.zeros((currsizeK, currsizeH), dtype=np.float64)
    sendtemp1 = np.zeros((currsizeK, currsizeH), dtype=np.float64)
    sendtemp2 = np.zeros((currsizeK, currsizeH), dtype=np.float64)
    sendtemp3 = np.zeros((currsizeK, currsizeH), dtype=np.float64)
    sendtemp4 = np.zeros((currsizeK, currsizeH), dtype=np.float64)
    sendtemp5 = np.zeros((currsizeK, currsizeH), dtype=np.float64)

    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None
    rectemp5 = None

    if rank == 0:
        rectemp = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp1 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp2 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp3 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp4 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp5 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)


    for i in range(currsizeK):
        for j in range(currsizeH):
            sendtemp[i,j], sendtemp1[i,j], sendtemp2[i,j], sendtemp3[i,j], sendtemp4[i,j], sendtemp5[i,j] = SSSF_zero(currK[i,j],V, pyp0)

    sendcounts = np.array(comm.gather(sendtemp.shape[0]*sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0]*sendtemp1.shape[1], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0]*sendtemp2.shape[1], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0]*sendtemp3.shape[1], 0))
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0]*sendtemp4.shape[1], 0))
    sendcounts5 = np.array(comm.gather(sendtemp5.shape[0]*sendtemp5.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)

    return rectemp, rectemp1, rectemp2, rectemp3, rectemp4, rectemp5


def graph_SSSF_pi(pyp0, K, V, rank, size):

    comm = MPI.COMM_WORLD
    nK = len(K)

    sizeK, sizeH = factors(size, nK)

    nb = nK / sizeK
    nh = nK / sizeH

    rankK = rank % sizeK
    rankH = rank // sizeK

    leftK = int(rankK * nb)
    rightK = int((rankK + 1) * nb)
    currsizeK = rightK - leftK

    leftH = int(rankH * nh)
    rightH = int((rankH + 1) * nh)
    currsizeH = rightH - leftH

    currK = K[leftK:rightK, leftH:rightH, :]
    
    sendtemp = np.zeros((currsizeK, currsizeH), dtype=np.float64)
    sendtemp1 = np.zeros((currsizeK, currsizeH), dtype=np.float64)
    sendtemp2 = np.zeros((currsizeK, currsizeH), dtype=np.float64)
    sendtemp3 = np.zeros((currsizeK, currsizeH), dtype=np.float64)
    sendtemp4 = np.zeros((currsizeK, currsizeH), dtype=np.float64)
    sendtemp5 = np.zeros((currsizeK, currsizeH), dtype=np.float64)

    rectemp = None
    rectemp1 = None
    rectemp2 = None
    rectemp3 = None
    rectemp4 = None
    rectemp5 = None

    if rank == 0:
        rectemp = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp1 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp2 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp3 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp4 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp5 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)


    for i in range(currsizeK):
        for j in range(currsizeH):
            sendtemp[i,j], sendtemp1[i,j], sendtemp2[i,j], sendtemp3[i,j], sendtemp4[i,j], sendtemp5[i,j] = SSSF_pi(currK[i,j],V, pyp0)

    sendcounts = np.array(comm.gather(sendtemp.shape[0]*sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0]*sendtemp1.shape[1], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0]*sendtemp2.shape[1], 0))
    sendcounts3 = np.array(comm.gather(sendtemp3.shape[0]*sendtemp3.shape[1], 0))
    sendcounts4 = np.array(comm.gather(sendtemp4.shape[0]*sendtemp4.shape[1], 0))
    sendcounts5 = np.array(comm.gather(sendtemp5.shape[0]*sendtemp5.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)
    comm.Gatherv(sendbuf=sendtemp3, recvbuf=(rectemp3, sendcounts3), root=0)
    comm.Gatherv(sendbuf=sendtemp4, recvbuf=(rectemp4, sendcounts4), root=0)
    comm.Gatherv(sendbuf=sendtemp5, recvbuf=(rectemp5, sendcounts5), root=0)



    return rectemp, rectemp1, rectemp2, rectemp3, rectemp4, rectemp5


#endregion


#region Graphing
def DSSFgraph(A,B,D, py0s, filename):
    plt.pcolormesh(A,B,D)
    plt.ylabel(r'$\omega/J_{zz}$')
    py0s.graph_loweredge(False)
    py0s.graph_upperedge(False)
    plt.savefig(filename+".png")
    plt.clf()


def SSSFGraph(A,B,d1, filename):
    plt.pcolormesh(A,B, d1)
    plt.colorbar()
    plt.ylabel(r'$(0,0,L)$')
    plt.xlabel(r'$(H,H,0)$')
    plt.savefig(filename+".png")
    plt.clf()
#endregion


#region SSSF DSSF Admin


def DSSF(nE, Jxx, Jyy, Jzz, h,n, filename, BZres, tol, which, flux=np.zeros(4)):

    if which == 0:
        py0s = py0.zeroFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n)
    elif which == 1:
        py0s = pypi.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n)
    else:
        py0s = pygen.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)

    py0s.solvemeanfield()

    kk = np.concatenate((np.linspace(gGamma1, gX, len(GammaX)), np.linspace(gX, gW1, len(XW)), np.linspace(gW1, gK, len(WK))
                         , np.linspace(gK,gGamma2, len(KGamma)), np.linspace(gGamma2, gL, len(GammaL)), np.linspace(gL, gU, len(LU)), np.linspace(gU, gW2, len(UW))))
    e = np.arange(py0s.TWOSPINON_GAP(kk)-0.5, py0s.TWOSPINON_MAX(kk)+0.5, nE)


    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if not pi:
        d1, d2, d3, d4 = graph_DSSF_zero(py0s, e, np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW)), tol, rank, size)
    else:
        d1, d2, d3, d4 = graph_DSSF_pi(py0s, e, np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW)), tol, rank, size)

    if rank == 0:

        f1 = "Files/"+filename+"_Szz_local"
        f2 = "Files/"+filename+"_Szz_global"
        f3 = "Files/"+filename+"_Sxx_local"
        f4 = "Files/"+filename+"_Sxx_global"
        np.savetxt(f1+".txt", d1)
        np.savetxt(f2+".txt", d2)
        np.savetxt(f3+".txt", d3)
        np.savetxt(f4+".txt", d4)
        X,Y = np.meshgrid(kk, e)
        DSSFgraph(X, Y, d1, py0s, f1)
        DSSFgraph(X, Y, d2, py0s, f2)
        DSSFgraph(X, Y, d3, py0s, f3)
        DSSFgraph(X, Y, d4, py0s, f4)
        # plt.show()

def samplegraph(nK, filenames):
    fig, axs = plt.subplots(3, len(filenames))

    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)


    # maxs = np.zeros(3)
    # mins = np.ones(3)*20
    # for i in range(len(filenames)):
    #     f1 = "Files/" + filenames[i] + "_local"
    #     f2 = "Files/" + filenames[i] + "_global"
    #     f3 = "Files/" + filenames[i] + "_NSF"
    #     d1 = np.loadtxt(f1+'.txt')
    #     d2 = np.loadtxt(f2 + '.txt')
    #     d3 = np.loadtxt(f3 + '.txt')
    #     d = [d1,d2,d3]

    for i in range(len(filenames)):
        f1 = "Files/" + filenames[i] + "_local"
        f2 = "Files/" + filenames[i] + "_global"
        f3 = "Files/" + filenames[i] + "_NSF"
        d1 = np.loadtxt(f1+'.txt')
        d2 = np.loadtxt(f2 + '.txt')
        d3 = np.loadtxt(f3 + '.txt')
        d = [d1/np.max(d1),d2/np.max(d2),d3/np.max(d3)]
        for j in range(3):
            axs[j, i].pcolormesh(A,B, d[j])
            axs[j, i].set_ylabel(r'$(0,0,L)$')
            axs[j, i].set_xlabel(r'$(H,H,0)$')
    plt.show()

def SSSF(nK, Jxx, Jyy, Jzz, h, n, v, BZres, filename, pi, flux=np.zeros(4)):
    if pi == 0:
        py0s = py0.zeroFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n)
    elif pi == 1:
        py0s = pypi.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n)
    else:
        py0s = pygen.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, flux=flux)

    py0s.solvemeanfield()
    
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hkltoK(A, B)

    
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if not pi:
        d1, d2, d3, d4, d5, d6 = graph_SSSF_zero(py0s, K, v, rank, size)
    else:
        d1, d2, d3, d4, d5, d6 = graph_SSSF_pi(py0s, K, v, rank, size)

    if rank == 0:
        f1 = "Files/" + filename + "Szz_local"
        f2 = "Files/" + filename + "Szz_global"
        f3 = "Files/" + filename + "Szz_NSF"
        f4 = "Files/" + filename + "Sxx_local"
        f5 = "Files/" + filename + "Sxx_global"
        f6 = "Files/" + filename + "Sxx_NSF"
        np.savetxt(f1 + '.txt', d1)
        np.savetxt(f2 + '.txt', d2)
        np.savetxt(f3 + '.txt', d3)
        np.savetxt(f4 + '.txt', d4)
        np.savetxt(f5 + '.txt', d5)
        np.savetxt(f6 + '.txt', d6)
        SSSFGraph(A, B, d1, f1)
        SSSFGraph(A, B, d2, f2)
        SSSFGraph(A, B, d3, f3)
        SSSFGraph(A, B, d4, f4)
        SSSFGraph(A, B, d5, f5)
        SSSFGraph(A, B, d6, f6)
#endregion

#region two spinon continuum
def TWOSPINCON(nK, h, n, Jxx, Jyy, Jzz, BZres, filename):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # if Jpm >= 0:
    #     py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, h=h, n=n, kappa=1)
    # else:

    py0s = pypi.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, kappa=2)
    py0s.solvemeanfield()

    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hkltoK(A, B).reshape((nK*nK,3))

    n = len(K)/size
    left = int(rank*n)
    right = int((rank+1)*n)

    currK = K[left:right, :]

    sendbuf1 = py0s.minMaxCal(currK)

    recvbuf1 = None

    if rank == 0:
        recvbuf1 = np.zeros((nK*nK,2))

    
    sendcounts = np.array(comm.gather(sendbuf1.shape[0]*sendbuf1.shape[1], 0))

    comm.Gatherv(sendbuf=sendbuf1, recvbuf=(recvbuf1, sendcounts), root=0)

    if rank == 0:

        f1 = "Files/" + filename + "_lower"
        f2 = "Files/" + filename + "_upper"
        loweredge = recvbuf1[:,0]
        upperedge = recvbuf1[:,1]
        loweredge = loweredge.reshape((nK, nK))
        upperedge = upperedge.reshape((nK, nK))
        np.savetxt(f1 + '.txt', loweredge)
        np.savetxt(f2 + '.txt', upperedge)
        # d1 = np.loadtxt(f1+'.txt')
        # d2 = np.loadtxt(f2 + '.txt')
        TWOSPINONGRAPH(A, B, loweredge, f1)
        TWOSPINONGRAPH(A, B, upperedge, f2)


def TWOSPINCON_0(nK, h, n, Jxx, Jyy, Jzz, BZres, filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # if Jpm >= 0:
    #     py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, h=h, n=n, kappa=1)
    # else:

    py0s = py0.zeroFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, kappa=2)
    py0s.solvemeanfield()

    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hkltoK(A, B).reshape((nK * nK, 3))

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

    py0s = pygen.piFluxSolver(Jxx, Jyy, Jzz, BZres=BZres, h=h, n=n, kappa=2, flux=flux)
    py0s.solvemeanfield()

    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hkltoK(A, B).reshape((nK * nK, 3))

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


def TWOSPINCON_gang(nK, h, n, Jpm, BZres, filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    py0s = pygang.piFluxSolver(Jpm, BZres=BZres, h=h, n=n, kappa=1)

    py0s.findLambda()

    H = np.linspace(0, 1, nK)
    L = np.linspace(0, 1, nK)
    A, B = np.meshgrid(H, L)
    K = twospinon_gangchen(A, B).reshape((nK * nK, 3))

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


def TWOSPINONGRAPH(A,B,d1, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.plot_surface(A,B, d1, cmap=cm.coolwarm)
    plt.savefig(filename+".png")
    plt.clf()
    plt.pcolormesh(A, B, d1)
    plt.savefig(filename+"_flatmesh.png")
    plt.clf()
#endregion

#endregion









