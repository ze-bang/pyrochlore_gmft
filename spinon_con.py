
from pyrochlore_dispersion_pi import green_pi, green_pi_branch, green_pi_old
from pyrochlore_dispersion import green_zero_branch, green_zero
from misc_helper import *
import matplotlib.pyplot as plt
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi



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

def DSSF_zero(q, omega, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks-q

    tempE= pyp0.E_zero(Ks)
    tempQ = pyp0.E_zero(Qs)


    greenp1 = green_zero_branch(Ks, pyp0)
    greenp2 = green_zero_branch(Qs, pyp0)

    #region S+- and S-+
    deltapm = delta(tempE, tempQ, omega,tol)
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)

    greenA = contract('ia, ib, iab->i',greenp1[:, :, 0,0], greenp2[:,:, 1,1], deltapm)
    greenpm = contract('i,ijk->ijk',greenA, (ffactpm+np.conj(np.transpose(ffactpm, (0,2,1)))))/4
    #endregion

    #region S++ and S--
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)

    greenB = contract('ia, ib, iab->i',greenp1[:, :, 0,1], greenp2[:,:, 1,0], deltapm)
    greenpp = contract('i,ijk->ijk',greenB, (ffactpp+np.conj(np.transpose(ffactpp, (0,2,1)))))/4

    S = (greenpp + greenpm)/2
    Sglobal = contract('ijk,jk->i', S, g(q))
    S = contract('ijk->i',S)

    return np.real(np.mean(S)), np.real(np.mean(Sglobal))


def DSSF_pi(q, omega, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks - q

    tempE = pyp0.LV_zero(Ks)[0]
    tempQ = pyp0.LV_zero(Qs)[0]


    greenpK = green_pi_branch(Ks, pyp0)
    greenpQ = green_pi_branch(Qs, pyp0)

    deltapm = delta(tempE, tempQ, omega, tol)
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)


    Spm = contract('ioab, ipyx, iop, abjk, jax, kby, ijk->ijk', greenpK[:,:,0:4,0:4], greenpQ[:,:,4:8,4:8], deltapm, A_pi_rs_rsp, piunitcell, piunitcell,
                    ffactpm)/4

    # print("we good")

    Smp = contract('ipba, ioxy, iop, abjk, jax, kby, ijk->ijk', greenpQ[:,:,0:4,0:4], greenpK[:,:,4:8,4:8], deltapm, A_pi_rs_rsp, piunitcell, piunitcell,
                    np.conj(ffactpm))/4

    Spp = contract('ioax, ipby, iop, abjk, jax, kby, ijk->ijk', greenpK[:,:,0:4,4:8], greenpQ[:,:,0:4, 4:8], deltapm, A_pi_rs_rsp_pp, piunitcell, piunitcell,
                    ffactpp)/4
    Smm = contract('ipxa, ioyb, iop, abjk, jax, kby, ijk->ijk', greenpQ[:,:,4:8,0:4], greenpK[:,:,4:8,0:4], deltapm, A_pi_rs_rsp_pp, piunitcell, piunitcell,
                    np.conj(ffactpp))/4


    S = (Spm + Smp + Spp + Smm)/4

    Sglobal = contract('ijk,jk->i', S, g(q))
    S = contract('ijk->i',S)


    return np.real(np.mean(S)), np.real(np.mean(Sglobal))

def graph_DSSF_zero(pyp0, E, K, tol, rank, size):
    # el = "==:==:=="
    # totaltask = len(E)*len(K)
    # increment = totaltask/50
    # count = 0

    comm = MPI.COMM_WORLD
    n = len(E)/size

    left = int(rank*n)
    right = int((rank+1)*n)
    
    currsize = right-left
    
    sendtemp = np.zeros((currsize, len(K)), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, len(K)), dtype=np.float64)

    currE = E[left:right]

    rectemp = None
    rectemp1 = None

    if rank == 0:
        rectemp = np.zeros((len(E), len(K)), dtype=np.float64)
        rectemp1 = np.zeros((len(E), len(K)), dtype=np.float64)


    for i in range(currsize):
        for j in range(len(K)):
            # start = time.time()
            # count = count + 1
            sendtemp[i,j], sendtemp1[i,j] = DSSF_zero(K[j], currE[i], pyp0, tol)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            # end = time.time()
            # el = (end - start)*(totaltask-count)
            # el = telltime(el)
            # sys.stdout.write('\r')
            # sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            # sys.stdout.flush()

    sendcounts = np.array(comm.gather(sendtemp.shape[0]*sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0]*sendtemp1.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)

    # print(rank)

    # if not MPI.Is_finalized():
    #     MPI.Finalize()

    return rectemp, rectemp1

def graph_DSSF_pi(pyp0, E, K, tol, rank, size):
    # el = "==:==:=="
    # totaltask = len(E)*len(K)
    # increment = totaltask/50
    # count = 0

    comm = MPI.COMM_WORLD
    n = len(E)/size

    left = int(rank*n)
    right = int((rank+1)*n)
    
    currsize = right-left
    
    sendtemp = np.zeros((currsize, len(K)), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, len(K)), dtype=np.float64)

    currE = E[left:right]

    rectemp = None
    rectemp1 = None

    if rank == 0:
        rectemp = np.zeros((len(E), len(K)), dtype=np.float64)
        rectemp1 = np.zeros((len(E), len(K)), dtype=np.float64)

    for i in range(currsize):
        for j in range(len(K)):
            # start = time.time()
            # count = count + 1
            sendtemp[i,j], sendtemp1[i,j] = DSSF_pi(K[j], currE[i], pyp0, tol)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            # end = time.time()
            # el = (end - start)*(totaltask-count)
            # el = telltime(el)
            # sys.stdout.write('\r')
            # sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            # sys.stdout.flush()

    sendcounts = np.array(comm.gather(sendtemp.shape[0]*sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0]*sendtemp1.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)

    # print(rank)

    # if not MPI.Is_finalized():
    #     MPI.Finalize()

    return rectemp, rectemp1

#endregion



#region SSSF

def SSSF_zero(q, v, pyp0):
    Ks = pyp0.bigB
    Qs = Ks-q
    le = len(Ks)
    # sQ = contract('i,j->ij', np.ones(le), q)

    greenp1 = green_zero(Ks, pyp0)
    greenp2 = green_zero(Qs, pyp0)

    #region S+- and S-+
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)

    greenA = greenp1[:,0,0] * greenp2[:,1,1]
    greenpm = contract('i,ijk->ijk',greenA, (ffactpm+np.conj(np.transpose(ffactpm, (0,2,1)))))/4
    #endregion

    #region S++ and S--
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)

    greenB = greenp1[:,0,1] * greenp2[:,1,0]
    greenpp = contract('i,ijk->ijk',greenB, (ffactpp+np.conj(np.transpose(ffactpp, (0,2,1)))))/4

    S = (greenpp + greenpm)/2
    Sglobal = contract('ijk,jk->i',S, g(q))
    SNSF = contract('ijk,jk->i',S, gNSF(v))
    S = contract('ijk->i',S)

    return np.real(np.mean(S)), np.real(np.mean(Sglobal)), np.real(np.mean(SNSF))


def SSSF_pi(q, v, pyp0):
    Ks = pyp0.bigB
    Qs = Ks - q
    v = v / magnitude(v)
    le = len(Ks)

    greenpK = green_pi(Ks, pyp0)
    greenpQ = green_pi(Qs, pyp0)

    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)

    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)

    # Spm = contract('iab, iyx, abjk, jax, kby, ijk->ijk', greenpKA, greenpQB, A_pi_rs_rsp, piunitcell, piunitcell,
    #                 ffactpm)/4
    # Smp = contract('iba, ixy, abjk, jax, kby, ijk->ijk', greenpQA, greenpKB, A_pi_rs_rsp, piunitcell, piunitcell,
    #                 np.conj(ffactpm))/4
    #
    # S = (Spm + Smp ) / 4

    #a = rs, b = rsp, y=index2, x=index1
    #
    Spm = contract('iab, iyx, abjk, jax, kby, ijk->ijk', greenpK[:,0:4,0:4], greenpQ[:,4:8,4:8], A_pi_rs_rsp, piunitcell, piunitcell,
                    ffactpm)/4
    Smp = contract('iba, ixy, abjk, jax, kby, ijk->ijk', greenpQ[:,0:4,0:4], greenpK[:,4:8,4:8], A_pi_rs_rsp, piunitcell, piunitcell,
                    np.conj(ffactpm))/4

    Spp = contract('iay, ibx, abjk, jax, kby, ijk->ijk', greenpK[:,0:4,4:8], greenpQ[:,0:4, 4:8], A_pi_rs_rsp_pp, piunitcell, piunitcell,
                    ffactpp)/4
    Smm = contract('iya, ixb, abjk, jax, kby, ijk->ijk', greenpQ[:,4:8,0:4], greenpK[:,4:8,0:4], A_pi_rs_rsp_pp, piunitcell, piunitcell,
                    np.conj(ffactpp))/4

    # print(Spm + Smp)
    # print('------------------------------')
    # print(Spm + Smp + Spp + Smm)
    # print('******************************')
    S = (Spm + Smp + Spp + Smm)/4


    Sglobal = contract('ijk,jk->i',S, g(q))
    SNSF = contract('ijk,jk->i',S, gNSF(v))
    S = contract('ijk->i',S)

    return np.real(np.mean(S)), np.real(np.mean(Sglobal)), np.real(np.mean(SNSF))

# def SSSF_pi_dumb(q, v, pyp0):
#     Ks = pyp0.bigB
#     Qs = Ks-q
#     v = v/magnitude(v)
#     le = len(Ks)
#
#     greenpKA = green_pi_old(Ks, 0, pyp0)
#     greenpKB = np.conj(greenpKA)
#     greenpQA = green_pi_old(Qs, 0, pyp0)
#     greenpQB = np.conj(greenpQA)
#
#
#     temp = np.zeros(le, dtype=np.complex128)
#     temp1 = np.zeros(le, dtype=np.complex128)
#     temp2 = np.zeros(le, dtype=np.complex128)
#     for rs in range(4):
#         for rsp in range(4):
#             for i in range(4):
#                 for j in range(4):
#                     index1 = np.array(np.where(piunitcell[i, rs] == 1))[0, 0]
#                     index2 = np.array(np.where(piunitcell[j, rsp] == 1))[0, 0]
#
#                     Spm = greenpKA[:, rs, rsp] * greenpQB[:, index2, index1]\
#                             *np.exp(1j * np.dot(Ks-q/2, NN[i]-NN[j])) \
#                             *np.exp(1j * (A_pi[rs, i] - A_pi[rsp, j]))/4
#                     Smp = greenpQA[:, rsp, rs] * greenpKB[:, index1, index2]\
#                             *np.exp(-1j * np.dot(Ks-q/2, NN[i]-NN[j])) \
#                             *np.exp(-1j * (A_pi[rs, i] - A_pi[rsp, j]))/4
#
#                     temp += (Spm + Smp)/4
#                     temp1 += (Spm + Smp) * (np.dot(z[i], z[j]) - np.dot(z[i], q) * np.dot(z[j], q) / np.dot(q, q))/4
#                     temp2 += (Spm + Smp) * (np.dot(z[i], v) * np.dot(z[j], v))/4
#
#
#     return np.real(np.mean(temp)), np.real(np.mean(temp1)), np.real(np.mean(temp2))


def graph_SSSF_zero(pyp0, K, V, rank, size):
    # el = "==:==:=="
    # totaltask =  K.shape[0]*K.shape[1]
    # increment = totaltask/50
    # count = 0

    comm = MPI.COMM_WORLD
    n = len(K)/size

    left = int(rank*n)
    right = int((rank+1)*n)
    
    currsize = right-left
    
    sendtemp = np.zeros((currsize, K.shape[1]), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, K.shape[1]), dtype=np.float64)
    sendtemp2 = np.zeros((currsize, K.shape[1]), dtype=np.float64)

    currK = K[left:right, :, :]

    rectemp = None
    rectemp1 = None
    rectemp2 = None

    if rank == 0:
        rectemp = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp1 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp2 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)


    for i in range(currsize):
        for j in range(K.shape[1]):
            # start = time.time()
            # count = count + 1
            sendtemp[i,j], sendtemp1[i,j], sendtemp2[i,j] = SSSF_zero(currK[i,j],V, pyp0)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            # end = time.time()
            # el = (end - start)*(totaltask-count)
            # el = telltime(el)
            # sys.stdout.write('\r')
            # sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            # sys.stdout.flush()
        # print(sendtemp[i])

    sendcounts = np.array(comm.gather(sendtemp.shape[0]*sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0]*sendtemp1.shape[1], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0]*sendtemp2.shape[1], 0))

    # print("h-----------------------------------------")

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    # print(rank)

    # if not MPI.Is_finalized():
    #     MPI.Finalize()

    return rectemp, rectemp1, rectemp2
    # E, K = np.meshgrid(e, K)


def graph_SSSF_pi(pyp0, K, V, rank, size):
    # el = "==:==:=="
    # totaltask = K.shape[0]*K.shape[1]
    # increment = totaltask/50
    # count = 0

    comm = MPI.COMM_WORLD
    n = len(K)/size

    left = int(rank*n)
    right = int((rank+1)*n)

    currsize = right-left
    
    sendtemp = np.zeros((currsize, K.shape[1]), dtype=np.float64)
    sendtemp1 = np.zeros((currsize, K.shape[1]), dtype=np.float64)
    sendtemp2 = np.zeros((currsize, K.shape[1]), dtype=np.float64)

    currK = K[left:right, :, :]

    rectemp = None
    rectemp1 = None
    rectemp2 = None

    if rank == 0:
        rectemp = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp1 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)
        rectemp2 = np.zeros((K.shape[0], K.shape[1]), dtype=np.float64)



    for i in range(currsize):
        for j in range(K.shape[1]):
            # start = time.time()
            # count = count + 1
            sendtemp[i,j], sendtemp1[i,j], sendtemp2[i,j] = SSSF_pi(currK[i,j],V, pyp0)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            # end = time.time()
            # el = (end - start)*(totaltask-count)
            # el = telltime(el)
            # sys.stdout.write('\r')
            # sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            # sys.stdout.flush()

    sendcounts = np.array(comm.gather(sendtemp.shape[0]*sendtemp.shape[1], 0))
    sendcounts1 = np.array(comm.gather(sendtemp1.shape[0]*sendtemp1.shape[1], 0))
    sendcounts2 = np.array(comm.gather(sendtemp2.shape[0]*sendtemp2.shape[1], 0))

    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    comm.Gatherv(sendbuf=sendtemp1, recvbuf=(rectemp1, sendcounts1), root=0)
    comm.Gatherv(sendbuf=sendtemp2, recvbuf=(rectemp2, sendcounts2), root=0)

    # if not MPI.Is_finalized():
    #     MPI.Finalize()

    return rectemp, rectemp1, rectemp2
    # E, K = np.meshgrid(e, K)


#endregion


#region Graphing
def DSSFgraph(A,B,D, py0s, filename):
    plt.pcolormesh(A,B,D)
    plt.ylabel(r'$\omega/J_{zz}$')
    py0s.graph_loweredge(False)
    py0s.graph_upperedge(False)
    plt.savefig(filename+".png")


def SSSFGraph(A,B,d1, filename):
    plt.pcolormesh(A,B, d1)
    plt.ylabel(r'$(0,0,L)$')
    plt.xlabel(r'$(H,H,0)$')


    # GammaH = np.array([0, 0])
    # LH =  np.array([1, 1])/2
    # XH = np.array([0, 1])
    # KH = np.array([3/4,0])
    # UH = np.array([1 / 4, 1])
    # UpH = np.array([1 / 4, -1])
    #
    # plt.plot([0],[0], marker='o', color='k')
    # plt.plot(np.linspace(XH, UH, 2).T[0], np.linspace(XH, UH, 2).T[1], marker='o', color='k')
    # plt.plot(np.linspace(UH, LH, 2).T[0], np.linspace(UH, LH, 2).T[1], marker='o', color='k')
    # plt.plot(np.linspace(KH, LH, 2).T[0], np.linspace(KH, LH, 2).T[1], marker='o', color='k')
    # plt.plot(np.linspace(KH, UpH, 2).T[0], np.linspace(KH, UpH, 2).T[1], color='k')
    # plt.plot(np.linspace(-UH, UpH, 2).T[0], np.linspace(-UH, UpH, 2).T[1], color='k')
    # plt.plot(np.linspace(-UH, -KH, 2).T[0], np.linspace(-UH, -KH, 2).T[1], color='k')
    # plt.plot(np.linspace(-UpH, -KH, 2).T[0], np.linspace(-UpH, -KH, 2).T[1], color='k')
    # plt.plot(np.linspace(-UpH, UH, 2).T[0], np.linspace(-UpH, UH, 2).T[1], color='k')
    # plt.text(GammaH[0]+0.03,GammaH[1]+0.03, '$\Gamma$')
    # plt.text(LH[0]+0.03,LH[1]+0.03, '$L$')
    # plt.text(XH[0]+0.03,XH[1]+0.03, '$X$')
    # plt.text(KH[0]+0.03,KH[1]+0.03, '$K$')
    # plt.text(UH[0] + 0.03, UH[1] + 0.03, '$U$')
    plt.savefig(filename+".png")

#endregion


#region SSSF DSSF Admin


def DSSF(nE, h,n,Jpm, filename, BZres, tol):
    if Jpm >= 0:
        py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, h=h, n=n)
    else:
        py0s = pypi.piFluxSolver(Jpm, BZres=BZres, h=h, n=n)

    py0s.findLambda()

    kk = np.concatenate((np.linspace(gGamma1, gX, len(GammaX)), np.linspace(gX, gW1, len(XW)), np.linspace(gW1, gK, len(WK))
                         , np.linspace(gK,gGamma2, len(KGamma)), np.linspace(gGamma2, gL, len(GammaL)), np.linspace(gL, gU, len(LU)), np.linspace(gU, gW2, len(UW))))
    e = np.arange(0, py0s.TWOSPINON_MAX(kk)+0.1, nE)


    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if Jpm >= 0:
        d1, d2 = graph_DSSF_zero(py0s, e, np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW)), tol, rank, size)
    else:
        d1, d2 = graph_DSSF_pi(py0s, e, np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW)), tol, rank, size)

    if rank == 0:
        f1 = "Files/"+filename+"_local"
        f2 = "Files/"+filename+"_global"
        np.savetxt(f1+".txt", d1)
        np.savetxt(f2+".txt", d2)
        # d1 = np.loadtxt("Files/"+filename+".txt")

        X,Y = np.meshgrid(kk, e)
        DSSFgraph(X, Y, d1, py0s, f1)
        DSSFgraph(X, Y, d2, py0s, f2)
        # plt.show()

def SSSF(nK, h, n, v, Jpm, BZres, filename):
    if Jpm >= 0:
        py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, h=h, n=n)
    else:
        py0s = pypi.piFluxSolver(Jpm, BZres=BZres, h=h, n=n)

    py0s.findLambda()

    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hkltoK(A, B)

    
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if Jpm >= 0:
        d1, d2, d3 = graph_SSSF_zero(py0s, K, v, rank, size)
    else:
        d1, d2, d3 = graph_SSSF_pi(py0s, K, v, rank, size)

    if rank == 0:
        f1 = "Files/" + filename + "_local"
        f2 = "Files/" + filename + "_global"
        f3 = "Files/" + filename + "_NSF"
        np.savetxt(f1 + '.txt', d1)
        np.savetxt(f2 + '.txt', d2)
        np.savetxt(f3 + '.txt', d3)
        # d1 = np.loadtxt(f1+'.txt')
        # d2 = np.loadtxt(f2 + '.txt')
        # d3 = np.loadtxt(f3 + '.txt')
        SSSFGraph(A, B, d1, f1)
        SSSFGraph(A, B, d2, f2)
        SSSFGraph(A, B, d3, f3)
#endregion

#region two spinon continuum
def TWOSPINCON(nK, h, n, Jpm, BZres, filename):
    if Jpm >= 0:
        py0s = py0.zeroFluxSolver(Jpm, BZres=BZres, h=h, n=n)
    else:
        py0s = pypi.piFluxSolver(Jpm, BZres=BZres, h=h, n=n)

    py0s.findLambda()

    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hkltoK(A, B).reshape((nK*nK,3))

    lower = py0s.minCal(K).reshape((nK, nK))
    upper = py0s.maxCal(K).reshape((nK, nK))

    f1 = "Files/" + filename + "_lower"
    f2 = "Files/" + filename + "_upper"
    np.savetxt(f1 + '.txt', lower)
    np.savetxt(f2 + '.txt', upper)
    # d1 = np.loadtxt(f1+'.txt')
    # d2 = np.loadtxt(f2 + '.txt')
    # d3 = np.loadtxt(f3 + '.txt')
    SSSFGraph(A, B, lower, f1)
    SSSFGraph(A, B, upper, f2)

#endregion









