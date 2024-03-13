import matplotlib.pyplot as plt
import numpy as np
from opt_einsum import contract
from numba import njit
from mpi4py import MPI
from misc_helper import hhltoK, hkztoK, genBZ
from archive.spinon_con import SSSFGraph
import h5py
import os

#Pyrochlore with XXZ Heisenberg on local coordinates. Couple of ways we can do this, simply project global cartesian coordinate
#onto local axis to determine local Sx, Sy, Sz.


z = np.array([[1,1,1],[1,-1,-1],[-1,1,-1], [-1,-1,1]])/np.sqrt(3)
y = np.array([[0,-1,1],[0,1,-1],[0,-1,-1], [0,1,1]])/np.sqrt(2)
x = np.array([[-2,1,1],[-2,-1,-1],[2,1,-1], [2,-1,1]])/np.sqrt(6)

coord = np.array([x,y,z])


#configuration follows the index of unit cell (x, y, z), unit cell index, spin components



@njit(cache=True)
def initialize_con(d):
    return get_random_spin((d,d,d,4))

@njit(cache=True)
def get_random_spin(shape):
    phi = np.random.uniform(0, 2*np.pi, shape)
    theta = np.arccos(np.random.uniform(-1, 1, shape))
    temp = np.zeros(shape+(3,))
    temp[..., 0] = np.cos(phi) * np.sin(theta)
    temp[..., 1] = np.sin(phi) * np.sin(theta)
    temp[..., 2] = np.cos(theta)
    return temp

@njit(cache=True)
def get_random_spin_single():
    phi = np.random.uniform(0, 2*np.pi)
    theta = np.arccos(np.random.uniform(-1, 1))
    temp = np.zeros(3)
    temp[0] = np.cos(phi) * np.sin(theta)
    temp[1] = np.sin(phi) * np.sin(theta)
    temp[2] = np.cos(theta)
    return temp


# ijk is the same displacement as in Felix's paper r1 r2 r3
# for u = 0, the nearest bond (aside from the same unit cell), comes from
#   i-1, u = 1; j-1, u = 2, k-2, u = 3
# for u = 1,
#   i+1, u = 0; (i+1, j-1, k), u = 2; (i+1, j, k-1), u = 3
# for u = 2,
#   (i, j+1, k), u = 0; (i-1, j+1, k), u = 1; (i, j+1, k-1), u = 3
# for u = 3,
#   (i, j, k+1), u = 0; (i-1, j, k+1), u = 1; (i, j-1, k+1), u = 2

# nearest neighbour based on
@njit(cache=True)
def indices(i,j,k,u,d):
    if u == 0:
        return np.array([[np.mod(i-1, d), j, k, 1], [i, np.mod(j-1, d), k, 2], [i, j, np.mod(k-1, d), 3]])
    if u == 1:
        return np.array([[np.mod(i+1, d), j, k, 0],[np.mod(i+1, d), np.mod(j-1, d), k, 2], [np.mod(i+1, d), j, np.mod(k-1, d), 3]])
    if u == 2:
        return np.array([[i, np.mod(j+1, d), k, 0], [np.mod(i-1, d), np.mod(j+1, d), k, 1], [i, np.mod(j+1, d), np.mod(k-1, d), 3]])
    if u == 3:
        return np.array([[i, j, np.mod(k+1, d), 0], [np.mod(i-1, d), j, np.mod(k+1, d), 1], [i, np.mod(j-1, d), np.mod(k+1, d), 2]])

@njit(cache=True)
def project(s, u):
    temp = np.zeros(3)
    temp[0] = np.dot(s, x[u])
    temp[1] = np.dot(s, y[u])
    temp[2] = np.dot(s, z[u])
    return temp

@njit(cache=True)
def localdot(k1, k2, Jxx, Jyy, Jzz):
    a, b, c = k1
    d, e, f = k2
    return Jxx*a*d+ Jyy*b*e + Jzz*c*f
@njit(cache=True)
def dot(k1, k2):
    a, b, c = k1
    d, e, f = k2
    return  (a*d+b*e) +c*f

@njit(fastmath=True, cache=True)
def energy_single_site_NN(con, i, j, k, u, Jxx, Jyy, Jzz):
    sum = 0.0
    for v in range(4):
        if not v == u:
            sum += localdot(con[i,j,k,u], con[i,j,k, v], Jxx, Jyy, Jzz)
    ind = indices(i,j,k,u, con.shape[0])
    for g in ind:
        sum += localdot(con[i,j,k,u], con[g[0], g[1], g[2],g[3]], Jxx, Jyy, Jzz)
    return sum/2

@njit(cache=True)
def energy_single_site(con, Jxx, Jyy, Jzz, gx, gy, gz, h, n, i, j, k, u):
    mag = dot(x[u], h*n) * gx * con[i,j,k,u,0] + dot(z[u], h*n) *gz * con[i,j,k,u,2] + dot(y[u], h*n) *gy * h**3 * (n[1]**3-3*n[0]**2*n[1]) *con[i,j,k,u,1]
    energy = energy_single_site_NN(con, i, j, k, u, Jxx, Jyy, Jzz)
    return energy - mag

@njit(cache=True)
def NN_field(con, i, j, k, u, Jxx, Jyy, Jzz):
    sum = np.zeros(3)
    for v in range(4):
        if not v == u:
            temp = np.copy(con[i,j,k,v])
            temp[0] = Jxx * temp[0]
            temp[1] = Jyy * temp[1]
            temp[2] = Jzz * temp[2]
            sum += temp

    ind = indices(i,j,k,u, con.shape[0])
    for g in ind:
        temp = np.copy(con[g[0], g[1], g[2],g[3]])
        temp[0] = Jxx * temp[0]
        temp[1] = Jyy * temp[1]
        temp[2] = Jzz * temp[2]
        sum += temp
    return sum/2

@njit(cache=True)
def get_deterministic_angle(con, Jxx, Jyy, Jzz, gx, gy, gz, h, n, i, j, k, u):
    temp = NN_field(con, i, j, k, u, Jxx, Jyy, Jzz)
    # print(temp)
    mag = dot(z[u], h*n) * (np.array([gx,0,gz])) + gy * h**3 * (n[1]**3-3*n[0]**2*n[1]) * np.array([0,1,0])
    temp = -temp + mag
    if not np.linalg.norm(temp) == 0:
        return temp/np.linalg.norm(temp)
    else:
        return temp

@njit(cache=True, fastmath=True)
def single_sweep(con, n, d, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec, T):
    enconold = 0.0
    for i in range(n):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    for s in range(4):
                        temp = np.copy(con[j,k,l,s])
                        con[j,k,l,s] = get_random_spin_single()
                        new = energy_single_site(con, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec, j, k, l, s)
                        deltaE = new - enconold
                        # print(enconold, new, deltaE, con[j,k,l,s], temp, s)
                        if deltaE < 0 or np.random.uniform(0,1) < np.exp(-deltaE/T):
                            enconold = new
                        else:
                            con[j,k,l,s] = temp
    return 0

@njit(fastmath=True, cache=True)
def deterministic_sweep(con, n, d, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec):
    for i in range(n):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    for s in range(4):
                        con[j,k,l,s] = get_deterministic_angle(con, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec, j, k, l, s)

    return 0

@njit(cache=True)
def annealing_schedule(x):
    return np.exp(-x)


@njit(cache=True)
def anneal(d, Target, Tinit, ntemp, nsweep, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec):
    con = initialize_con(d)
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()

    x = Tinit*np.logspace(0, Target, ntemp)

    for i in x:
        print(i)
        single_sweep(con, nsweep, d, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec, i)
    deterministic_sweep(con, 1000000, d, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec)
    return con

r = np.array([[0,1/2,1/2],[1/2,0,1/2],[1/2,1/2,0]])

NN = np.array([[-1/4,-1/4,-1/4],[-1/4,1/4,1/4],[1/4,-1/4,1/4],[1/4,1/4,-1/4]])/4



def magnetization(con):
    mag = contract('ijkus->s', con)
    return mag/(con.shape[0]**3*4)/2
@njit
def realcoords(con):
    d = con.shape[0]
    rcoord = np.zeros(con.shape)
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for x in range(4):
                    rcoord[a,b,c,x] = a*r[0]+b*r[1]+c*r[2]+NN[x]
    return rcoord


def spin_q_e(con, rcoord, q):
    d = con.shape[0]
    ffact = np.exp(1j*contract('ik, xyzuk->ixyzu',q,rcoord))
    S = contract('ixyzu, xyzus->is',ffact,con)
    S = S / np.sqrt(d ** 3 * 4)
    return S

# @njit(cache=True)
def SSSF_q_e(con, rcoord, q):
    A = spin_q_e(con, rcoord, q)
    B = spin_q_e(con, rcoord, -q)
    return np.real(contract('ia, ib->iab',A,B))


def spin_q(con, rcoord, q):
    d = con.shape[0]
    ffact = np.exp(1j*contract('ijk, xyzuk->ijxyzu',q,rcoord))
    S = contract('ijxyzu, xyzus->ijs',ffact,con)
    S = S / np.sqrt(d ** 3 * 4)
    return S

# @njit(cache=True)
def SSSF_q(con, rcoord, q):
    A = spin_q(con, rcoord, q)
    B = spin_q(con, rcoord, -q)
    return np.real(contract('ija, ijb->abij',A,B))


def SSSF(con, rcoord, nK, filename, n):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    if (n == np.array([0,0,1])).all():
        K = hkztoK(A, B)
    else:
        K = hhltoK(A, B)
    K = K.reshape((nK*nK,3))
    S = SSSF_q_e(con, rcoord, K)
    f1 = filename + "Sxx_local"
    f2 = filename + "Syy_local"
    f3 = filename + "Szz_local"
    f4 = filename + "Sxy_local"
    f5 = filename + "Sxz_local"
    f6 = filename + "Syz_local"
    S = contract('ijab->abij',S.reshape((nK, nK,3,3)))

    np.savetxt(f1 + '.txt', S[0,0])
    np.savetxt(f2 + '.txt', S[1,1])
    np.savetxt(f3 + '.txt', S[2,2])
    np.savetxt(f4 + '.txt', S[0,1])
    np.savetxt(f5 + '.txt', S[0,2])
    np.savetxt(f6 + '.txt', S[1,2])
    SSSFGraph(A, B, S[0,0], f1)
    SSSFGraph(A, B, S[1,1], f2)
    SSSFGraph(A, B, S[2,2], f3)
    SSSFGraph(A, B, S[0, 1], f4)
    SSSFGraph(A, B, S[0, 2], f5)
    SSSFGraph(A, B, S[1, 2], f6)

BasisBZA = np.array([2*np.pi*np.array([-1,1,1]),2*np.pi*np.array([1,-1,1]),2*np.pi*np.array([1,1,-1])])

# @njit(cache=True)
def ordering_q_slice(con, rcoord, ind):
    K = genBZ(101)
    S = np.abs(SSSF_q_e(con, rcoord, K))
    Szz = S[:,ind,ind]
    max = np.max(Szz)
    if max < 1e-13:
        qzz = np.array([np.NaN, np.NaN, np.NaN])
    else:
        indzz = np.array([])
        tempindzz = np.where(Szz==max)[0]
        indzz = np.concatenate((indzz, tempindzz))
        indzz = np.array(indzz.flatten(),dtype=int)
        qzz = np.unique(np.mod(K[indzz],1),axis=0)
    if qzz.shape == (3,):
        qzz = qzz.reshape(1,3)
    qzz = contract('ij,jk->ik', qzz, BasisBZA)
    return qzz

def ordering_q(con, rcoord):
    temp = np.concatenate((ordering_q_slice(con, rcoord, 0),ordering_q_slice(con, rcoord, 1),ordering_q_slice(con, rcoord, 2)))
    return temp


def plottetrahedron(x,y,z, ax):
    center = x*r[0]+y*r[1]+z*r[2]
    coords = center + NN
    start = np.zeros((6,3))
    start[0] = start[1] = start[2] = coords[0]
    start[3] = start[4] = coords[1]
    start[5] = coords[2]
    end = np.zeros((6,3))
    end[0] = coords[1]
    end[1] = end[3] = coords[2]
    end[2] = end[4] = end[5] = coords[3]

    for i in range(6):
        ax.plot([start[i,0], end[i,0]], [start[i,1], end[i,1]], zs=[start[i,2], end[i,2]], color='blue')

def graphconfig(con):
    d = con.shape[0]
    ax = plt.axes(projection='3d')
    coord = np.zeros((4*d*d*d,3))
    spin = np.zeros((4*d*d*d,3))
    d = con.shape[0]
    for i in range(d):
        for j in range(d):
            for k in range(d):
                plottetrahedron(i,j,k,ax)
                for u in range(4):
                    coord[i*d*d*4+j*d*4+k*4+u] = i*r[0]+j*r[1]+k*r[2]+NN[u]
                    spin[i * d * d * 4 + j * d * 4 + k * 4 + u] = con[i,j,k,u,0]*x[u]+con[i,j,k,u,1]*y[u]+con[i,j,k,u,2]*z[u]

    spin = spin*0.5

    ax.scatter(coord[:,0], coord[:,1], coord[:,2])
    ax.quiver(coord[:,0], coord[:,1], coord[:,2],spin[:,0], spin[:,1], spin[:,2], color='red', length=0.3)
    plt.savefig("test_monte_carlo.png")
    plt.show()

# @njit(cache=True, fastmath=True)
def phase_diagram(nK, sites, nT, nSweep, h, hvec, filename):
    Jx = np.linspace(-1, 1, nK)
    Jz = np.linspace(-1, 1, nK)
    phase = np.zeros((nK,nK))
    tol = 1e-6

    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nb = nK/size

    left = int(rank*nb)
    right = int((rank+1)*nb)
    currsize = right-left

    currJx = Jx[left:right]

    sendtemp = np.zeros((currsize, nK), dtype=np.float64)
    rectemp = None
    if rank == 0:
        rectemp = np.zeros((nK, nK), dtype=np.float64)

    for i in range(currsize):
        for j in range(nK):
            con = np.copy(anneal(sites, -10, 1, nT, nSweep, currJx[i], 1, Jz[j], 0.01, 4e-4, 1, h, hvec))
            mag = abs(magnetization(con))
            if mag[0] > tol:
                sendtemp[i,j] = 0
            elif mag[2] > tol:
                sendtemp[i,j] = 1
            elif (mag<tol).all():
                if currJx[i] + Jz[j] > 0:
                    sendtemp[i,j] = 2
                else:
                    sendtemp[i,j] = 3

    sendcounts = np.array(comm.gather(sendtemp.shape[0] * sendtemp.shape[1], 0))
    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)

    if rank == 0:
        np.savetxt(filename+'.txt', rectemp)
        plt.contourf(Jx, Jz, rectemp.T)
        plt.xlabel(r'$J_x$')
        plt.ylabel(r'$J_z$')
        plt.savefig(filename+ '.png')
        plt.clf()
    return 0

h111= np.array([1,1,1])/np.sqrt(3)
h110 = np.array([1,1,0])/np.sqrt(2)
h001 = np.array([0,0,1])

#gx = 0.01, gy = 4e-4, gz=1

def monte_SSSF(filename, Jxx, Jyy, Jzz, h, n, gx, gy, gz, d, Target, Tinit, ntemp, nsweep):
    con = np.copy(anneal(d, Target, Tinit, ntemp, nsweep, Jxx, Jyy, Jzz, gx, gy, gz, h, n))
    rcoord = realcoords(con)
    f = h5py.File(filename+".h5", 'w')
    f.create_dataset("spins", data=con)
    f.create_dataset("positions", data=rcoord)
    f.close()
    A = ordering_q(con, rcoord)
    np.savetxt(filename+"ordering_q.txt", A)
    M = magnetization(con)
    np.savetxt(filename+"magnetization.txt",M)
    SSSF(con, rcoord, 100, filename, n)

def scan_line(dirname, Jxx, Jyy, Jzz, hmin, hmax, nScan, n, gx, gy, gz, d, Target, Tinit, ntemp, nsweep):
    hs = np.linspace(hmin, hmax,nScan)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    dirString = ""
    if (n==np.array([0,0,1])).all():
        dirString = "001"
    elif (n==np.array([1,1,0])/np.sqrt(2)).all():
        dirString = "110"
    else:
        dirString = "111"

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nb = nScan/size

    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK
    currH = hs[leftK:rightK]


    for i in range(currsizeK):
        filename = dirname+"h_"+dirString+"="+str(currH[i])+"/"
        if not os.path.isdir(filename):
            os.mkdir(filename)
        monte_SSSF(filename, Jxx, Jyy, Jzz, currH[i], n, gx, gy, gz, d, Target, Tinit, ntemp, nsweep)

def scan_all(n, L):
    scan_line('./monte_carlo_files/Jpm_0.3/', -0.6, 1, -0.6, 0, 1, 20, n, 0, 0, 1, L, -7, 1, 200, 100000)
    scan_line('./monte_carlo_files/Jpm_-0.3/', 0.6, 1, 0.6, 0, 1, 20, n, 0, 0, 1, L, -7, 1, 200, 100000)
    scan_line('./monte_carlo_files/Jpm_0.1/', -0.2, 1, -0.2, 0, 1, 20, n, 0, 0, 1, L, -7, 1, 200, 100000)
    scan_line('./monte_carlo_files/Jpm_-0.1/', 0.2, 1, 0.2, 0, 1, 20, n, 0, 0, 1, L, -7, 1, 200, 100000)





