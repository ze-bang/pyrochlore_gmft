import matplotlib.pyplot as plt
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
    ind = np.mod(indices(i,j,k,u), con.shape[0])
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
            temp = con[i,j,k,v]
            sum += Jxx*temp[0]+ Jyy*temp[1] + Jzz*temp[2]

    ind = np.mod(indices(i,j,k,u), con.shape[0])
    for g in ind:
        temp = con[g[0], g[1], g[2],g[3]]
        sum += Jxx*temp[0] + Jyy*temp[1] + Jzz*temp[2]
    return sum/2

@njit(cache=True)
def get_deterministic_angle(con, Jxx, Jyy, Jzz, gx, gy, gz, h, n, i, j, k, u):
    temp = NN_field(con, i, j, k, u, Jxx, Jyy, Jzz)
    mag = dot(z[u], h*n) * (gx * np.array([1,0,0]) + gz * np.array([0,0,1])) + gy * h**3 * (n[1]**3-3*n[0]**2*n[1]) * np.array([0,1,0])
    temp = temp - mag
    if not np.linalg.norm(temp) == 0:
        return temp/np.linalg.norm(temp)
    else:
        return temp


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
@njit(fastmath=True, cache=True)
def single_sweep(con, n, d, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec, T):
    enconold = 0.0
    for i in range(n):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    for s in range(4):
                        oldcon = con
                        con[j,k,l,s] = get_random_spin_single()
                        new = energy_single_site(con, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec, j, k, l, s)
                        deltaE = new - enconold
                        if deltaE < 0 or np.random.uniform(0,1) < np.exp(-deltaE/T):
                            enconold = new
                        else:
                            con = oldcon
    return con

@njit(fastmath=True, cache=True)
def deterministic_sweep(con, n, d, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec):
    for i in range(n):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    for s in range(4):
                        con[j,k,l,s] = get_deterministic_angle(con, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec, j, k, l, s)
    return con

@njit(cache=True)
def annealing_schedule(x):
    return np.exp(-x)


@njit(cache=True)
def anneal(d, Target, Tinit, ntemp, nsweep, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec):
    con = initialize_con(d)
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()

    x = np.linspace(1, Target, ntemp)
    T = Tinit*annealing_schedule(x)
    temp = np.zeros(3, dtype=np.float64)

    for i in T:
        if i > 1e-7:
            temp = single_sweep(con, nsweep, d, Jxx, Jyy, Jzz, gx, gy, gz, h, hvec, i)
        else:
            temp = deterministic_sweep(con, nsweep, d, Jxx, Jyy, gx, gy, gz, Jzz, h, hvec)
        con = temp
        # print(con)
    return con

r = np.array([[0,1/2,1/2],[1/2,0,1/2],[1/2,1/2,0]])*2

b = np.array([[-1/4,-1/4,-1/4],[-1/4,1/4,1/4],[1/4,-1/4,1/4],[1/4,1/4,-1/4]])

@njit(cache=True, fastmath=True)
def magnetization(con):
    d = con.shape[0]
    mag = np.zeros(3, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for u in range(4):
                    mag += con[i,j,k,u]
    mag = mag/(d**3)

    return mag
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
                    coord[i*d*d*4+j*d*4+k*4+u] = i*r[0]+j*r[1]+k*r[2]+b[u]
                    spin[i * d * d * 4 + j * d * 4 + k * 4 + u] = con[i,j,k,u,0]*x[u]+con[i,j,k,u,1]*y[u]+con[i,j,k,u,2]*z[u]

    spin = spin*0.5

    ax.scatter(coord[:,0], coord[:,1], coord[:,2])
    ax.quiver(coord[:,0], coord[:,1], coord[:,2],spin[:,0], spin[:,1], spin[:,2])
    plt.savefig("test_monte_carlo.png")
    plt.show()

@njit(cache=True, fastmath=True)
def phase_diagram():
    Jx = np.linspace(-1, 1, 50)
    Jz = np.linspace(-1, 1, 50)
    phase = np.zeros((50,50))
    tol = 1e-3
    for i in range(50):
        for j in range(50):
            con = anneal(4, 100, 1, int(1e3), int(1e5), Jx[i], 1, Jz[j], 0, np.array([0,0,1]))
            mag = magnetization(con)
            if mag[0] > tol:
                phase[i,j] = 0
            elif mag[2] > tol:
                phase[i,j] = 1
            elif (mag<tol).all():
                if Jx[i] + Jz[j] > 0:
                    phase[i,j] = 2
                else:
                    phase[i,j] = 3
    return phase

con = anneal(4, 100, 1, int(1e2), int(1e4), 0, -1, 1, 0.01, 4e-4, 1, 0, np.array([0,0,1]))
print(magnetization(con))
con = anneal(4, 100, 1, int(1e2), int(1e4), -1, 0, 1, 0.01, 4e-4, 1, 0, np.array([0,0,1]))
print(magnetization(con))
con = anneal(4, 100, 1, int(1e2), int(1e4), 0, -0.2, 1, 0.01, 4e-4, 1, 0, np.array([0,0,1]))
print(magnetization(con))
con = anneal(4, 100, 1, int(1e2), int(1e4), 0, 0.2, 1, 0.01, 4e-4, 1, 0, np.array([0,0,1]))
print(magnetization(con))


# phase = phase_diagram()
# np.savetxt('phase_monte_carlo.txt', phase)
# plt.contourf(phase)
# plt.savefig('phase_monte_carlo.png')
# plt.show()






