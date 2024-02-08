import matplotlib.pyplot as plt
import numpy as np
from opt_einsum import contract
from numba import njit
from mpi4py import MPI
from misc_helper import hkltoK, genBZ
from archive.spinon_con import SSSFGraph

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
    mag = dot(z[u], h*n) * (gx * con[i,j,k,u,0] + gz * con[i,j,k,u,2]) + gy * h**3 * (n[1]**3-3*n[0]**2*n[1]) *con[i,j,k,u,1]
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

    x = Tinit*np.logspace(1, Target, ntemp)

    for i in x:
        if i > 1e-8:
            single_sweep(con, nsweep, d, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec, i)
        else:
            deterministic_sweep(con, nsweep, d, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec)
        # con = temp
        # print(con)
        # print("------------------------------------------------")
    return con

r = np.array([[0,1/2,1/2],[1/2,0,1/2],[1/2,1/2,0]])*2

NN = np.array([[-1/4,-1/4,-1/4],[-1/4,1/4,1/4],[1/4,-1/4,1/4],[1/4,1/4,-1/4]])



def magnetization(con):
    mag = contract('ijkus->s', con)
    return mag/(con.shape[0]**3*4)

@njit(cache=True)
def Sab(k1, k2):
    a, b, c = k1
    d, e, f = k2
    return np.array([[a*d, a*e, a*f],[b*d, b*e, b*f],[c*d, c*e, c*f]])

@njit(cache=True)
def SSSF_q(con, q):
    d = con.shape[0]
    # print(d)
    S = np.zeros((3,3), dtype=np.complex128)
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for x in range(4):
                    for e in range(d):
                        for f in range(d):
                            for g in range(d):
                                for h in range(4):
                                    temp = Sab(con[a,b,c,x], con[e,f,g,h])
                                    temp = np.exp(1j*np.dot(q,(a*r[0]+b*r[1]+c*r[2]+NN[x] - e*r[0]+f*r[1]+g*r[2]+NN[h])))*temp
                                    # if (temp == np.nan).any() or (temp > 1e5).any():
                                    S += temp
    S = S / (d ** 3 * 4)
    # print(S.shape)
    # print('------------------------------------')
    return S


def SSSF(con, nK, filename):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hkltoK(A, B)
    S = np.zeros((K.shape[0],K.shape[1],3,3))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            S[i,j] = SSSF_q(con, K[i,j])
    f1 = "Files/" + filename + "Sxx_local"
    f2 = "Files/" + filename + "Syy_local"
    f3 = "Files/" + filename + "Szz_local"
    f4 = "Files/" + filename + "Sxy_local"
    f5 = "Files/" + filename + "Sxz_local"
    f6 = "Files/" + filename + "Syz_local"

    np.savetxt(f1 + '.txt', S[:,:,0,0])
    np.savetxt(f2 + '.txt', S[:,:,1,1])
    np.savetxt(f3 + '.txt', S[:,:,2,2])
    np.savetxt(f4 + '.txt', S[:,:,0,1])
    np.savetxt(f5 + '.txt', S[:,:,0,2])
    np.savetxt(f6 + '.txt', S[:,:,1,2])
    # d1 = np.loadtxt(f1+'.txt')
    # d2 = np.loadtxt(f2 + '.txt')
    # d3 = np.loadtxt(f3 + '.txt')
    SSSFGraph(A, B, S[:,:,0,0], f1)
    SSSFGraph(A, B, S[:,:,1,1], f2)
    SSSFGraph(A, B, S[:,:,2,2], f3)
    SSSFGraph(A, B, S[:, :, 0, 1], f4)
    SSSFGraph(A, B, S[:, :, 0, 2], f5)
    SSSFGraph(A, B, S[:, :, 1, 2], f6)

# @njit(cache=True)
def ordering_q(con):
    K = genBZ(100)
    S = np.zeros((K.shape[0], 3, 3))
    q = np.zeros((3, 3, 3))
    for i in range(K.shape[0]):
        S[i] = SSSF_q(con, K[i])

    ind = np.argmax(S, axis=0)

    for i in range(3):
        for j in range(3):
            q[i,j] = K[ind[i,j],:]

    return q


def graphconfig(con):
    d = con.shape[0]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    coord = np.zeros((4*d*d*d,3))
    spin = np.zeros((4*d*d*d,3))
    d = con.shape[0]
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for u in range(4):
                    coord[i*d*d*4+j*d*4+k*4+u] = i*r[0]+j*r[1]+k*r[2]+NN[u]
                    spin[i * d * d * 4 + j * d * 4 + k * 4 + u] = con[i,j,k,u,0]*x[u]+con[i,j,k,u,1]*y[u]+con[i,j,k,u,2]*z[u]

    spin = spin*0.5

    ax.scatter(coord[:,0], coord[:,1], coord[:,2])
    ax.quiver(coord[:,0], coord[:,1], coord[:,2],spin[:,0], spin[:,1], spin[:,2])
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
        np.savetxt('Files/' + filename+'.txt', rectemp)
        plt.contourf(Jx, Jz, rectemp.T)
        plt.xlabel(r'$J_x$')
        plt.ylabel(r'$J_z$')
        plt.savefig('Files/' + filename+ '.png')
        plt.clf()
    return 0



# phase = phase_diagram(25, 1, int(1e2), int(1e4), 0, np.array([0,0,1]), 'phase_monte_carlo')

con = np.copy(anneal(8, -10, 1, 100, 10000, 0, 0, 1, 0.01, 4e-4, 1, 2, np.array([0,0,1])))
print(con)
q = ordering_q(con)
print(q)

# np.savetxt('phase_monte_carlo.txt', phase)
# plt.contourf(phase)
# plt.savefig('phase_monte_carlo.png')
# plt.show()






