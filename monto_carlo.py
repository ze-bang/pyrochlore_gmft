import numpy as np
from opt_einsum import contract
from numba import njit, jit
from mpi4py import MPI

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
        return np.array([[i-1, j, k, 1], [i, j-1, k, 2], [i, j, k-1, 3]])
    if u == 1:
        return np.array([[i+1, j, k, 0],[i+1, j-1, k, 2], [i+1, j, k-1, 3]])
    if u == 2:
        return np.array([[i, j+1, k, 0], [i-1, j+1, k, 1], [i, j+1, k-1, 3]])
    if u == 3:
        return np.array([[i, j, k+1, 0], [i-1, j, k+1, 1], [i, j-1, k+1, 2]])

@njit(cache=True)
def project(s, u):
    temp = np.zeros(3)
    temp[0] = np.dot(s, x[u])
    temp[1] = np.dot(s, y[u])
    temp[2] = np.dot(s, z[u])
    return temp

@njit(cache=True)
def localdot(k1, k2, Jzz, Jxy):
    a, b, c = k1
    d, e, f = k2
    return  Jxy*(a*d+b*e) + Jzz*c*f
@njit(cache=True)
def dot(k1, k2):
    a, b, c = k1
    d, e, f = k2
    return  (a*d+b*e) +c*f

@njit(fastmath=True, cache=True)
def energy_single_site_NN(con, i, j, k, u, Jzz, Jxy):
    sum = 0.0
    for v in range(4):
        if not v == u:
            sum += localdot(project(con[i,j,k,u],u), project(con[i,j,k, v],v), Jzz, Jxy)
    ind = indices(i,j,k,u,con.shape[1])
    for g in ind:
        sum += localdot(project(con[i,j,k,u], u), project(con[g[0], g[1], g[2],g[3]], g[3]), Jzz, Jxy)
    return sum/2

@njit( cache=True)
def energy_single_site(con, Jzz, Jxy, h, n, i, j, k, u):
    mag = dot(con[i,j,k, u], -h*n)
    energy = energy_single_site_NN(con, i, j, k, u, Jzz, Jxy)
    return energy+mag

# @njit()
# def numba_subrountine_energy(d, con, confast):
#     energy = 0.0
#     for i in range(d):
#         for j in range(d):
#             for k in range(d):
#                 for u in range(4):
#                     energy += energy_single_site_NN(con, confast, i, j, k, u)
#     return energy
#
# def energy(con, Jzz, Jxy, h, n):
#     conlocal = contract('ijkus, xus->ijkux', con, coord)
#
#     confast = np.zeros(con.shape)
#     confast[0:2] = Jxy*conlocal[0:2]
#     confast[2] = Jzz*conlocal[2]
#
#     d = con.shape[1]
#
#     mag = np.sum(contract('ijkus, s', con, -h * n))
#     energy = numba_subrountine_energy(d, con, confast)
#
#     return (energy+mag)/(d**3)
# @njit(fastmath=True, cache=True)
def sweep(con, n, d, Jzz, Jxy, h, hvec, T):
    enconold = 0.0
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nb = d / size

    left = int(rank * nb)
    right = int((rank + 1) * nb)
    currsize = right-left
    currd = np.zeros((currsize+2, d,d,4,3), dtype=np.float64)

    if rank == 0:
        currd[0] = con[-1]
        currd[1:-1] = con[left:right+1]
    elif rank == size-1:
        currd[0:currsize+1] = con[left-1:right]
        currd[-1] = con[0]
    else:
        currd = con[left-1:right+1]

    recvbuf = None

    if rank == 0:
        recvbuf = np.zeros((d, d, d, 4, 3), dtype=np.float64)

    for i in range(n):
        for j in range(left, right):
            for k in range(d):
                for l in range(d):
                    for s in range(4):
                        oldcon = con
                        currd[j,k,l,s] = get_random_spin_single()
                        deltaE = energy_single_site(currd, Jzz, Jxy, h, hvec, j, k, l, s) - enconold
                        if deltaE > 0 or np.random.uniform(0,1) > np.exp(deltaE/T):
                            currd = oldcon
                        else:
                            enconold = enconold + deltaE

                        destleft = rank - 1
                        destright = rank + 1
                        if rank == 0:
                            destleft = size-1
                        if rank == size:
                            destright = 0
                        comm.Send([currd[1], MPI.FLOAT], dest=destleft)
                        comm.Send([currd[-2], MPI.FLOAT], dest=destright)

                        boundright = np.empty((d, d, 4, 3), dtype=np.float64)
                        boundleft = np.empty((d, d, 4, 3), dtype=np.float64)

                        comm.Recv([boundright, MPI.FLOAT], source=destright)
                        comm.Recv([boundleft, MPI.FLOAT], source=destleft)

                        currd[0] = boundleft
                        currd[-1] = boundright

    sendcounts = np.array(comm.gather(currd.shape[0] * d**2 * 4 * 3, 0))

    comm.Gatherv(sendbuf=currd, recvbuf=(recvbuf, sendcounts), root=0)

    return recvbuf
@njit(cache=True)
def annealing_schedule(x):
    return np.exp(-x)


# @njit(cache=True)
def anneal(d, Target, Tinit, n, Jzz, Jxy, h, hvec):
    T = Tinit
    con = initialize_con(d)
    x=1.0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    while T > Target:
        temp = sweep(con, n, d, Jzz, Jxy, h, hvec, T)
        if rank == 0:
            con = temp
            T = Tinit*annealing_schedule(x)
            x += 0.01
    return con


con = anneal(4, 1e-7, 1, int(1e8), 1, 1, 0, np.array([0,0,1]))
np.savetxt('spin_config_Jzz=1_Jxy=1_h=0.txt', con)




