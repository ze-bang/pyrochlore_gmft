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
def indices(i,j,k,u):
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
    return Jxy*(a*d+b*e) + Jzz*c*f
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
    ind = indices(i,j,k,u)
    for g in ind:
        sum += localdot(project(con[i,j,k,u], u), project(con[g[0], g[1], g[2],g[3]], g[3]), Jzz, Jxy)
    return sum/2

@njit(cache=True)
def energy_single_site(con, Jzz, Jxy, h, n, i, j, k, u):
    mag = dot(con[i,j,k, u], -h*n)
    energy = energy_single_site_NN(con, i, j, k, u, Jzz, Jxy)
    return energy+mag

@njit(cache=True)
def NN_field(con, i, j, k, u, Jzz, Jxy):
    sum = np.zeros(3)
    for v in range(4):
        if not v == u:
            temp = project(con[i,j,k, v],v)
            sum += Jxy*(temp[0]+temp[1]) + Jzz*temp[2]

    ind = indices(i,j,k,u)
    for g in ind:
        temp = project(con[g[0], g[1], g[2],g[3]], g[3])
        sum += Jxy * (temp[0] + temp[1]) + Jzz * temp[2]
    return sum/2

def get_deterministic_angle(con, Jzz, Jxy, h, n, i, j, k, u):
    return NN_field(con, Jzz, Jxy, h, n, i, j, k, u) - h * n

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

def single_sweep(con, n, d, Jzz, Jxy, h, hvec, T):
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
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    for s in range(4):
                        oldcon = currd
                        currd[j,k,l,s] = get_random_spin_single()
                        deltaE = enconold - energy_single_site(currd, Jzz, Jxy, h, hvec, j, k, l, s)
                        if deltaE < 0 or np.random.uniform(0,1) > np.exp(-deltaE/T):
                            enconold = enconold + deltaE
                        else:
                            currd = oldcon

                        destleft = np.mod(rank - 1, size)
                        destright = np.mod(rank + 1, size)

                        sendleft = np.array(currd[0], dtype=np.float64)
                        sendright = np.array(currd[-1], dtype=np.float64)

                        comm.Send(sendleft, destleft, tag=77)
                        comm.Send(sendright, destright, tag=80)

                        boundleft = np.zeros((d,d,4,3), dtype=np.float64)
                        boundright = np.zeros((d,d,4,3), dtype=np.float64)

                        comm.Recv(boundleft, source=destleft, tag=80)
                        comm.Recv(boundright, source=destright, tag=77)

                        currd[0] = boundleft
                        currd[-1] = boundright


    sendcounts = np.array(comm.gather(currsize * d * d * 4 *3, 0))
    comm.Gatherv(sendbuf=currd[1:-2], recvbuf=(recvbuf, sendcounts), root=0)
    return recvbuf


def deterministic_sweep(con, n, d, Jzz, Jxy, h, hvec, T):
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
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    for s in range(4):

                        currd[j,k,l,s] = get_deterministic_angle(currd, Jzz, Jxy, h, hvec, j, k, l, s)

                        destleft = np.mod(rank - 1, size)
                        destright = np.mod(rank + 1, size)

                        sendleft = np.array(currd[0], dtype=np.float64)
                        sendright = np.array(currd[-1], dtype=np.float64)

                        comm.Send(sendleft, destleft, tag=77)
                        comm.Send(sendright, destright, tag=80)

                        boundleft = np.zeros((d,d,4,3), dtype=np.float64)
                        boundright = np.zeros((d,d,4,3), dtype=np.float64)

                        comm.Recv(boundleft, source=destleft, tag=80)
                        comm.Recv(boundright, source=destright, tag=77)

                        currd[0] = boundleft
                        currd[-1] = boundright


    sendcounts = np.array(comm.gather(currsize * d * d * 4 *3, 0))
    comm.Gatherv(sendbuf=currd[1:-2], recvbuf=(recvbuf, sendcounts), root=0)
    return recvbuf

@njit(cache=True)
def annealing_schedule(x):
    return np.exp(-x)


# @njit(cache=True)
def anneal(d, Target, Tinit, ntemp, nsweep, Jzz, Jxy, h, hvec):
    con = initialize_con(d)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    x = np.linspace(1, Target, ntemp)
    T = Tinit*annealing_schedule(x)

    for i in T:
        if T > 1e-6:
            temp = single_sweep(con, nsweep, d, Jzz, Jxy, h, hvec, i)
        else:
            temp = deterministic_sweep(con, nsweep, d, Jzz, Jxy, h, hvec, i)
        if rank == 0:
            con = temp
    return con


con = anneal(4, 100, 1, int(1e2),int(5e3), 1, 1, 0, np.array([0,0,1]))
np.savetxt('spin_config_Jzz=1_Jxy=1_h=0.txt', con)






