import numpy as np
from itertools import permutations
import math
import numba as nb
from opt_einsum import contract
import math
from mpi4py import MPI
from functools import reduce
from scipy.optimize import minimize, root_scalar

def factors(n, nK):
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0 and n / i <= nK:
            return n / i, i

graphres=60

chi_A = np.array([[[0,1,1,1],
                  [1,0,-1,-1],
                  [1,-1,0,-1],
                  [1,-1,-1,0]],
                  [[0, -1, 1, 1],
                   [-1, 0, 1, 1],
                   [1, 1, 0, -1],
                   [1, 1, -1, 0]],
                  [[0, -1, -1, 1],
                   [-1, 0, -1, 1],
                   [-1, -1, 0, 1],
                   [1, -1, 1, 0]],
                  [[0, 1, -1, 1],
                   [1, 0, 1, -1],
                   [-1, 1, 0, 1],
                   [1, -1, 1, 0]]
                  ])

xipicell = np.array([[[1,1,1,1],[1,-1,1,1],[1,-1,-1,1],[1,1,-1,1]],[[1,-1,-1,-1],[1,1,-1,-1],[1,1,1,-1],[1,-1,1,-1]]])

@nb.njit
def magnitude(vector):
    temp = 0
    for i in vector:
        temp = temp + i**2
    return np.sqrt(temp)

e0 = np.array([0,0,0])
e1 = np.array([0,1,1])/2
e2 = np.array([1,0,1])/2
e3 = np.array([1,1,0])/2


b0 = np.pi * np.array([1, 1, 1])
b1 = np.pi * np.array([-1, 1, 1])
b2 = np.pi * np.array([1, -1, 1])
b3 = np.pi * np.array([1, 1, -1])

Gamma = np.array([0, 0, 0])

L = np.pi * np.array([1, 1, 1])
X = 2 * np.pi * np.array([1, 0, 0])
W = 2 * np.pi * np.array([1, 0, 1 / 2])
K = 2 * np.pi * np.array([3/4, 0, 3 / 4])
U = 2 * np.pi * np.array([1, 1/4, 1 / 4])

stepN = magnitude(np.abs(U-W))/graphres


@nb.njit
def repcoord(a, b, c):
    return a*b1+b*b2+c*b3


@nb.njit
def realcoord(r):
    r1, r2, r3 = r
    return r1*e1 +r2*e2 + r3* e3


z = np.array([np.array([1,1,1])/np.sqrt(3), np.array([1,-1,-1])/np.sqrt(3), np.array([-1,1,-1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)])
@nb.njit
def BasisBZ(mu):
    if mu == 0:
        return 2*np.pi*np.array([-1,1,1])
    if mu == 1:
        return 2*np.pi*np.array([1,-1,1])
    if mu == 2:
        return 2*np.pi*np.array([1,1,-1])

BasisBZA = np.array([2*np.pi*np.array([-1,1,1]),2*np.pi*np.array([1,-1,1]),2*np.pi*np.array([1,1,-1])])

@nb.njit
def neta(alpha):
    if alpha == 0:
        return 1
    else:
        return -1

def telltime(sec):
    hours = math.floor(sec/3600)
    sec = sec-hours*3600
    minus = math.floor(sec/60)
    sec = int(sec - minus * 60)
    return str(hours) + ':' + str(minus) + ':' + str(sec)


def gaussian(mu, tol):
    return np.exp(-np.power( - mu, 2) / (2 * np.power(tol, 2)))

def cauchy(mu, tol):
    return tol/(mu**2+tol**2)/np.pi

def bose(beta, omega):
    if beta == 0:
        return np.zeros(omega.shape)
    else:
        return 1/(np.exp(beta*omega)-1)

def g(q):
    M = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if not np.dot(q,q) == 0:
                M[i,j] = np.dot(z[i], z[j]) - np.dot(z[i],q) * np.dot(z[j],q)/ np.dot(q,q)
            else:
                M[i, j] = np.dot(z[i], z[j])
    return M

def gNSF(v):
    M = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            M[i,j] = np.dot(z[i], v) * np.dot(z[j], v)
    return M

NN = np.array([np.array([-1 / 4, -1 / 4, -1 / 4]), np.array([-1 / 4, 1 / 4, 1 / 4]), np.array([1 / 4, -1 / 4, 1 / 4]), np.array([1 / 4, 1 / 4, -1 / 4])])


NNminus = np.zeros((4,4,3), dtype=float)
for i in range(4):
    for j in range(4):
        NNminus[i,j,:] = NN[i]-NN[j]


NNplus = np.zeros((4,4,3), dtype=float)
for i in range(4):
    for j in range(4):
        NNplus[i,j,:] = NN[i]+NN[j]

def genmask(b):
    a = np.sum(np.where((b==0)|(b==1), 1, 0))
    if a == 3:
        return 1/8
    if a == 2:
        return 1/4
    if a == 1:
        return 1/2
    else:
        return 1

def applymask(b):
    return np.apply_along_axis(genmask, 1, b)


def setdiff3d(a1, a2):
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    return np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])

def indextoignore(a1, a2):
    a1_temp = np.copy(a1)
    a2_temp = np.copy(a2)
    a1_rows = a1_temp.view([('', np.float64)] * a1_temp.shape[1])
    a2_rows = a2_temp.view([('', np.float64)] * a2_temp.shape[1])
    return np.array(np.where(np.in1d(a1_rows, a2_rows)==True)[0], dtype=int)

# @nb.njit
def genBZ(d, m=1):
    dj = d*1j
    b = np.mgrid[0:m:dj, 0:m:dj, 0:m:dj].reshape(3,-1).T
    b = np.concatenate((b,genALLSymPointsBare()))
    temp = np.array(contract('ij, jk->ik', b, BasisBZA), order='C')
    return temp, b


# def genBZfolded(d):
#     d = d*1j
#     b = np.mgrid[0:1:d, 0:1:d, 0:1:d].reshape(3,-1)
#     temp = contract('ij, ik->jk', b, BasisBZA)
#     return temp


@nb.njit
def step(mu):
    if mu == 0:
        return np.array([0,0,0])
    if mu == 1:
        return np.array([1,0,0])
    if mu == 2:
        return np.array([0,1,0])
    if mu == 3:
        return np.array([0,0,1])




A_pi = np.array([[0,0,0,0],
                  [0,np.pi,0,0],
                  [0,np.pi,np.pi,0],
                  [0,0,np.pi,0]])

# A_pi = np.zeros((4,4))

piunitcell = np.array([
    [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]],
    [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]],
    [[0,1,0,0],
     [1,0,0,0],
     [0,0,0,1],
     [0,0,1,0]],
    [[0,0,1,0],
     [0,0,0,1],
     [1,0,0,0],
     [0,1,0,0]]
])

number = 8
notrace = np.ones((4,4))-np.diag([1,1,1,1])

A_pi_rs_traced = np.zeros((4,4,4))


for i in range(4):
    for j in range(4):
        for k in range(4):
            A_pi_rs_traced[i,j,k] = np.real(np.exp(1j * (A_pi[i,j] - A_pi[i,k])))


A_pi_rs_traced_pp = np.zeros((4,4,4))


for i in range(4):
    for j in range(4):
        for k in range(4):
            A_pi_rs_traced_pp[i,j,k] = np.real(np.exp(1j * (A_pi[i,j] + A_pi[i,k])))


A_pi_rs_rsp = np.zeros((4,4,4,4))

for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                A_pi_rs_rsp[i,j,k,l] = np.real(np.exp(1j * (A_pi[i,k] - A_pi[j,l])))


A_pi_rs_rsp_pp = np.zeros((4,4,4,4))

for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                A_pi_rs_rsp_pp[i,j,k,l] = np.real(np.exp(1j * (A_pi[i,k] + A_pi[j,l])))


@nb.njit
def findS(r):
    for i in range (4):
        if np.all(r == unitCell(i)):
            return i
    return -1


@nb.njit
def unitCell(mu):
    if mu == 0:
        return np.array([0,0,0])
    if mu == 1:
        return np.array([0,1,0])
    if mu == 2:
        return np.array([0,0,1])
    if mu == 3:
        return np.array([0,1,1])


def drawLine(A, B, stepN):
    N = magnitude(np.abs(A-B))
    num = int(N/stepN)
    temp = np.linspace(A, B, num)
    return temp


@nb.njit
def NNtest(mu):
    if mu == 0:
        return np.array([0, 0, 0])/2
    if mu == 1:
        return np.array([0, 1, 1])/2
    if mu == 2:
        return np.array([1, 0, 1])/2
    if mu == 3:
        return np.array([1, 1, 0])/2

@nb.njit
def b(mu):
    if mu == 0:
        return b1
    if mu == 1:
        return b2
    if mu == 2:
        return b3



def obliqueProj(W):
    M = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            M[i][j] = np.dot(BasisBZA[i], BasisBZA[j])

    y = contract('j,ij->i', W, BasisBZA)
    A = np.array(np.matmul(np.linalg.inv(M), y))
    return A


GammaX = drawLine(Gamma, X, stepN)
XW = drawLine(X, W, stepN)
WK = drawLine(W, K, stepN)
KGamma = drawLine(K, Gamma, stepN)
GammaL = drawLine(Gamma, L, stepN)
LU = drawLine(L, U, stepN)
UW = drawLine(U, W, stepN)


gGamma1 = 0
gX = magnitude(np.abs(Gamma-X))
gW1 = gX + magnitude(np.abs(X-W))
gK = gW1 + magnitude(np.abs(W-K))
gGamma2 = gK + magnitude(np.abs(K-Gamma))
gL = gGamma2 + magnitude(np.abs(Gamma-L))
gU = gL + magnitude(np.abs(L-U))
gW2 = gU + magnitude(np.abs(U-W))

def msp(items):
  '''Yield the permutations of `items` where items is either a list
  of integers representing the actual items or a list of hashable items.
  The output are the unique permutations of the items given as a list
  of integers 0, ..., n-1 that represent the n unique elements in
  `items`.

  Examples
  ========

  # >>> for i in msp('xoxox'):
  # ...   print(i)

  [1, 1, 1, 0, 0]
  [0, 1, 1, 1, 0]
  [1, 0, 1, 1, 0]
  [1, 1, 0, 1, 0]
  [0, 1, 1, 0, 1]
  [1, 0, 1, 0, 1]
  [0, 1, 0, 1, 1]
  [0, 0, 1, 1, 1]
  [1, 0, 0, 1, 1]
  [1, 1, 0, 0, 1]

  Reference: "An O(1) Time Algorithm for Generating Multiset Permutations", Tadao Takaoka
  https://pdfs.semanticscholar.org/83b2/6f222e8648a7a0599309a40af21837a0264b.pdf
  '''
  def visit(head):
      (rv, j) = ([], head)
      for i in range(N):
          (dat, j) = E[j]
          rv.append(dat)
      return rv

  u = list(set(items))
  E = list(reversed(sorted([u.index(i) for i in items])))
  N = len(E)
  # put E into linked-list format
  (val, nxt) = (0, 1)
  for i in range(N):
      E[i] = [E[i], i + 1]
  E[-1][nxt] = None
  head = 0
  afteri = N - 1
  i = afteri - 1
  yield visit(head)
  while E[afteri][nxt] is not None or E[afteri][val] < E[head][val]:
      j = E[afteri][nxt]  # added to algorithm for clarity
      if j is not None and E[i][val] >= E[j][val]:
          beforek = afteri
      else:
          beforek = i
      k = E[beforek][nxt]
      E[beforek][nxt] = E[k][nxt]
      E[k][nxt] = head
      if E[k][val] < E[head][val]:
          i = k
      afteri = E[i][nxt]
      head = k
      yield visit(head)


def permute(G):
    B = []
    for i in msp(G):
        B += [i]
    return B

def genALLSymPointsBare():
    d = 9 * 1j
    b = np.mgrid[0:1:d, 0:1:d, 0:1:d].reshape(3, -1).T
    return b

def equi_class_c3(K1, K2):
    if (K1 == K2).all() or (K1 == np.array([K2[2], K2[0], K2[1]])).all() or (K1 == np.array([K2[1], K2[2], K2[0]])).all():
        return True
    else:
        return False
def symmetry_equivalence(K):
    L = len(K)
    dind = []
    keep = []
    for i in range(L):
        for j in range(L):
            if (not i == j) and equi_class_c3(K[i], K[j]) and (not j in keep):
                keep += [i]
                dind += [j]
    K = np.delete(K, dind, 0)
    return np.array(K)


def genALLSymPoints():
    d = 9 * 1j
    b = np.mgrid[-2*np.pi:2*np.pi:d, -2*np.pi:2*np.pi:d, -2*np.pi:2*np.pi:d].reshape(3, -1).T
    return b

symK = genALLSymPoints()

def phase0(lams, minLams, pi):
    lamA, lamB = lams
    if np.all(lams < minLams):
        print("AFM Phase")
        return 0
    elif lamA < minLams[0]:
        print("PMu Phase")
        return 1+pi
    elif lamB < minLams[1]:
        print("PMu Phase")
        return -(1+pi)
    else:
        print("QSL Phase")
        print(3*(-1)**pi)
        return 3*(-1)**pi


def phaseMag(lams, minLams, pi):
    lamA, lamB = lams
    if np.all(lams < minLams):
        print("AFM Phase")
        return 0
    elif lamA - minLams[0] < 0:
        print("PMu Phase")
        return 1+pi
    elif lamB - minLams[1]< 0:
        print("PMu Phase")
        return -(1+pi)
    else:
        print("QSL Phase")
        print(3*(-1)**pi)
        return 3*(-1)**pi

def BZbasis(mu):
    if mu == 0:
        return 2*np.pi*np.array([1,0,0])
    elif mu == 1:
        return 2*np.pi*np.array([0,1,0])
    elif mu == 2:
        return 2*np.pi*np.array([0,0,1])



def hkltoK(H, L):
    return np.einsum('ij,k->ijk',H, BZbasis(0)+BZbasis(1)) + np.einsum('ij,k->ijk',L, BZbasis(2))


def gangchen(mu):
    if mu == 0:
        return 2*np.pi*np.array([-1,1,1])
    elif mu == 1:
        return 2*np.pi*np.array([1,-1,1])



def twospinon_gangchen(H, L):
    return np.einsum('ij,k->ijk',H, gangchen(0)) + np.einsum('ij,k->ijk',L, gangchen(1))


h111=np.array([1,1,1])/np.sqrt(3)
h001=np.array([0,0,1])
h110 = np.array([1,1,0])/np.sqrt(2)
h1b10 = np.array([1,-1,0])/np.sqrt(2)
hb110 = np.array([-1,1,0])/np.sqrt(2)
h100 = np.array([1,0,0])

def beta1(t):
    return 0.9**t

def beta2(t):
    return 0.99**t


def size_non_h(A):
    sum = 0
    for i in range(len(A)):
        sum = sum + len(A[i])
    return sum

def non_h_unique(A):
    B = []
    for i in range(len(A)):
        unique = True
        for j in range(i+1, len(A)):
            if (A[i] == A[j]).all():
                unique = False
        if unique:
            B = B + [A[i]]
    return B

deltamin=100
def adaptive_simpsons_3D(f, x0, x1, y0, y1, z0, z1, tol, *args):
    def simpsons_core(f, x0, x1, y0, y1, z0, z1, *args):
        hx = (x1 - x0) / 2
        hy = (y1 - y0) / 2
        hz = (z1 - z0) / 2

        x_mid = x0 + hx
        y_mid = y0 + hy
        z_mid = z0 + hz

        integral = (hx * hy * hz / 27) * (
                f(x0, y0, z0, *args) + 4 * f(x_mid, y0, z0, *args) + f(x1, y0, z0, *args) +
                4 * f(x0, y_mid, z0, *args) + 16 * f(x_mid, y_mid, z0, *args) + 4 * f(x1, y_mid, z0, *args) +
                f(x0, y1, z0, *args) + 4 * f(x_mid, y1, z0, *args) + f(x1, y1, z0, *args) +
                4 * f(x0, y0, z_mid, *args) + 16 * f(x_mid, y0, z_mid, *args) + 4 * f(x1, y0, z_mid, *args) +
                16 * f(x0, y_mid, z_mid, *args) + 64 * f(x_mid, y_mid, z_mid, *args) + 16 * f(x1, y_mid, z_mid, *args) +
                4 * f(x0, y1, z_mid, *args) + 16 * f(x_mid, y1, z_mid, *args) + 4 * f(x1, y1, z_mid, *args) +
                f(x0, y0, z1, *args) + 4 * f(x_mid, y0, z1, *args) + f(x1, y0, z1, *args) +
                4 * f(x0, y_mid, z1, *args) + 16 * f(x_mid, y_mid, z1, *args) + 4 * f(x1, y_mid, z1, *args) +
                f(x0, y1, z1, *args) + 4 * f(x_mid, y1, z1, *args) + f(x1, y1, z1, *args)
        )
        return integral


    def adaptive_simpsons_aux(f, x0, x1, y0, y1, z0, z1, tol, integral_prev, *args):
        x_mid = (x0 + x1) / 2
        y_mid = (y0 + y1) / 2
        z_mid = (z0 + z1) / 2

        integral_left_bottom_front = simpsons_core(f, x0, x_mid, y0, y_mid, z0, z_mid, *args)
        integral_left_bottom_back = simpsons_core(f, x0, x_mid, y0, y_mid, z_mid, z1, *args)
        integral_left_top_front = simpsons_core(f, x0, x_mid, y_mid, y1, z0, z_mid, *args)
        integral_left_top_back = simpsons_core(f, x0, x_mid, y_mid, y1, z_mid, z1, *args)
        integral_right_bottom_front = simpsons_core(f, x_mid, x1, y0, y_mid, z0, z_mid, *args)
        integral_right_bottom_back = simpsons_core(f, x_mid, x1, y0, y_mid, z_mid, z1, *args)
        integral_right_top_front = simpsons_core(f, x_mid, x1, y_mid, y1, z0, z_mid, *args)
        integral_right_top_back = simpsons_core(f, x_mid, x1, y_mid, y1, z_mid, z1, *args)

        integral = (integral_left_bottom_front + integral_left_bottom_back +
                    integral_left_top_front + integral_left_top_back +
                    integral_right_bottom_front + integral_right_bottom_back +
                    integral_right_top_front + integral_right_top_back)

        error = abs(integral - integral_prev)
        # print(x_mid, y_mid, z_mid, integral, integral_prev, error < 15*tol)
        if error < 15 * tol:
            return integral + error / 15
        else:
            integral_left_bottom_front = adaptive_simpsons_aux(f, x0, x_mid, y0, y_mid, z0, z_mid, tol / 8,
                                                               integral_left_bottom_front, *args)
            integral_left_bottom_back = adaptive_simpsons_aux(f, x0, x_mid, y0, y_mid, z_mid, z1, tol / 8,
                                                              integral_left_bottom_back, *args)
            integral_left_top_front = adaptive_simpsons_aux(f, x0, x_mid, y_mid, y1, z0, z_mid, tol / 8,
                                                             integral_left_top_front, *args)
            integral_left_top_back = adaptive_simpsons_aux(f, x0, x_mid, y_mid, y1, z_mid, z1, tol / 8,
                                                            integral_left_top_back, *args)
            integral_right_bottom_front = adaptive_simpsons_aux(f, x_mid, x1, y0, y_mid, z0, z_mid, tol / 8,
                                                                integral_right_bottom_front, *args)
            integral_right_bottom_back = adaptive_simpsons_aux(f, x_mid, x1, y0, y_mid, z_mid, z1, tol / 8,
                                                               integral_right_bottom_back, *args)
            integral_right_top_front = adaptive_simpsons_aux(f, x_mid, x1, y_mid, y1, z0, z_mid, tol / 8,
                                                              integral_right_top_front, *args)
            integral_right_top_back = adaptive_simpsons_aux(f, x_mid, x1, y_mid, y1, z_mid, z1, tol / 8,
                                                             integral_right_top_back, *args)

            return (integral_left_bottom_front + integral_left_bottom_back +
                    integral_left_top_front + integral_left_top_back +
                    integral_right_bottom_front + integral_right_bottom_back +
                    integral_right_top_front + integral_right_top_back)

    integral_approx = adaptive_simpsons_aux(f, x0, x1, y0, y1, z0, z1, tol, 0, *args)
    return integral_approx


def gauss_quadrature_3D_pts(a, b, c, d, e, g, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)

    amp = np.array([(b-a)/2,(d-c)/2,(g-e)/2])
    # Map nodes from the interval [-1, 1] to the interval [a, b]
    gauss_pts = contract('k, ik->ik',amp,np.array(np.meshgrid(nodes, nodes, nodes)).reshape(3, -1).T) + np.array([(a+b)/2,(c+d)/2,(e+g)/2])
    weights = contract('i,j,k->ijk', weights, weights, weights).ravel()
    weights *= 0.125 * (b - a) * (d - c) * (g - e)
    return gauss_pts, weights
def integrate(f, gauss_pts, weights, *args):
    integral_approximation = np.dot(weights, f(gauss_pts, *args))
    return integral_approximation

def riemann_sum_3d_pts(a, b, c, d, p, q, n):
    dx = (b - a) / n
    dy = (d - c) / n
    dz = (q - p) / n
    xi = np.linspace(a,b,n)
    yj = np.linspace(c,d,n)
    zk = np.linspace(p,q,n)
    pts = np.array(np.meshgrid(xi, yj, zk)).reshape(3, -1).T
    return pts, np.ones(n**3)*dx*dy*dz

def monte_carlo_integration_3d_pts(a, b, c, d, p, q, n):
    volume = (b - a) * (d - c) * (q - p)

    x = np.random.uniform(a, b, n)
    y = np.random.uniform(c, d, n)
    z = np.random.uniform(p, q, n)

    pts = np.array(np.meshgrid(x, y, z)).reshape(3, -1).T
    return pts, np.ones(n**3)/(n**3)*volume

def simpsons_rule_3d_pts(a, b, c, d, p, q, n):
    dx = (b - a) / (2 * n)
    dy = (d - c) / (2 * n)
    dz = (q - p) / (2 * n)

    xi = np.linspace(a, b, 2 * n + 1)
    yj = np.linspace(c, d, 2 * n + 1)
    zk = np.linspace(p, q, 2 * n + 1)

    # Compute weights for each dimension
    weight_x = np.full_like(xi, 4)
    weight_x[::2] = 2
    weight_x[0] = weight_x[-1] = 1

    weight_y = np.full_like(yj, 4)
    weight_y[::2] = 2
    weight_y[0] = weight_y[-1] = 1

    weight_z = np.full_like(zk, 4)
    weight_z[::2] = 2
    weight_z[0] = weight_z[-1] = 1

    pts = np.array(np.meshgrid(xi, yj, zk)).reshape(3, -1).T
    weights = contract('i,j,k->ijk', weight_x, weight_y, weight_z).ravel()*dx*dy*dz/27

    return pts, weights

def trapezoidal_rule_3d_pts(a, b, c, d, p, q, n):
    xi = np.linspace(a, b, n + 1)
    yj = np.linspace(c, d, n + 1)
    zk = np.linspace(p, q, n + 1)

    dx = (b - a) / n
    dy = (d - c) / n
    dz = (q - p) / n

    weight_x = np.full_like(xi, 1)
    weight_x[0] = weight_x[-1] = 1/2

    weight_y = np.full_like(yj, 1)
    weight_y[0] = weight_y[-1] = 1/2

    weight_z = np.full_like(zk, 1)
    weight_z[0] = weight_z[-1] = 1/2

    pts = np.array(np.meshgrid(xi, yj, zk)).reshape(3, -1).T
    weights = contract('i,j,k->ijk', weight_x, weight_y, weight_z).ravel()*dx*dy*dz
    return pts, weights

def adaptive_gauss_quadrature_3d(f, a, b, c, d, e, g, tol, *args, max_depth=10, n=3):
    """
    Adaptive Gaussian quadrature for integrating a function over a 3D region.

    Parameters:
    - f: Function to integrate.
    - a, b, c, d, e, g: Integration limits in the x, y, and z directions.
    - tol: Tolerance for accuracy.
    - max_depth: Maximum recursion depth.
    - n: Number of nodes (and weights).

    Returns:
    - Integral approximation.
    """
    def adaptively_integrate(x0, x1, y0, y1, z0, z1, n, depth, last, *args):
        """
        Recursive function to adaptively integrate over the region.
        """
        if depth == 0:
            integral_approximation = gauss_quadrature_3d(f, x0, x1, y0, y1, z0, z1, n, *args)
        else:
            integral_approximation = last
        x_mid = 0.5 * (x0 + x1)
        y_mid = 0.5 * (y0 + y1)
        z_mid = 0.5 * (z0 + z1)

        subregions = [
            (x0, x_mid, y0, y_mid, z0, z_mid),
            (x_mid, x1, y0, y_mid, z0, z_mid),
            (x0, x_mid, y_mid, y1, z0, z_mid),
            (x_mid, x1, y_mid, y1, z0, z_mid),
            (x0, x_mid, y0, y_mid, z_mid, z1),
            (x_mid, x1, y0, y_mid, z_mid, z1),
            (x0, x_mid, y_mid, y1, z_mid, z1),
            (x_mid, x1, y_mid, y1, z_mid, z1)
        ]
        # Compute the integral approximation for the four subregions
        subregion_integrals = np.array([gauss_quadrature_3d(f, *subregion, n, *args) for subregion in subregions])

        # Compute the error
        error = np.abs(np.sum(subregion_integrals) - integral_approximation)

        if error < tol or depth >= max_depth:
            return integral_approximation

        integral = 0.0
        for i in range(8):
            integral += adaptively_integrate(*subregions[i], n, depth + 1, subregion_integrals[i], *args)

        return integral

    # Define the initial subregion
    return adaptively_integrate(a, b, c, d, e, g, n, 0, 0, *args)
