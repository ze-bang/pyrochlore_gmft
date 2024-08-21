import time

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from opt_einsum import contract
import math
from flux_stuff import *
# import pyrochlore_exclusive_boson as pyeb
# import matplotlib.pyplot as plt
def factors(n, nK):
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0 and n / i <= nK:
            return n / i, i

graphres=25

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
BasisBZA = np.array([2*np.pi*np.array([-1,1,1]),2*np.pi*np.array([1,-1,1]),2*np.pi*np.array([1,1,-1])])
def magnitude(vector):
    temp = np.einsum('i,ik->k', vector, BasisBZA)
    return np.linalg.norm(temp)

def magnitude_bi(vector1, vector2):
    # temp1 = np.einsum('i,ik->k', vector1, BasisBZA)
    # temp2 = np.einsum('i,ik->k', vector2, BasisBZA)
    temp1 = vector1
    temp2 = vector2
    return np.linalg.norm(temp1-temp2)

e0 = np.array([0,0,0])
e1 = np.array([0,1,1])/2
e2 = np.array([1,0,1])/2
e3 = np.array([1,1,0])/2


b0 = np.pi * np.array([1, 1, 1])
b1 = np.pi * np.array([-1, 1, 1])
b2 = np.pi * np.array([1, -1, 1])
b3 = np.pi * np.array([1, 1, -1])

zppz = np.array([0, np.pi, np.pi, 0])
pzzp = np.array([np.pi, 0, 0, np.pi])

# Gamma = np.array([0, 0, 0])
# L = np.pi * np.array([1, 1, 1])
# X = 2 * np.pi * np.array([1, 0, 0])
# W = 2 * np.pi * np.array([1, 0, 1 / 2])
# K = 2 * np.pi * np.array([3/4, 0, 3 / 4])
# U = 2 * np.pi * np.array([1, 1/4, 1 / 4])


# Gamma = np.array([0, 0, 0])
# L = np.array([1, 1, 1])/2
# X = np.array([0, 0.5, 0.5])
# W = np.array([0.25, 0.75, 0.5])
# K = np.array([0.375, 0.75, 0.375])
# U = np.array([0.25, 0.625, 0.625])

Gamma = np.array([0, 0, 0])
K = 2 * np.pi * np.array([3/4, -3/4, 0])
W = 2 * np.pi * np.array([1, -1/2, 0])
X = 2 * np.pi * np.array([1, 0, 0])

L = np.pi * np.array([1, 1, 1])
U = 2 * np.pi * np.array([1/4, 1/4, 1])
W1 = 2 * np.pi * np.array([0, 1/2, 1])
X1 = 2 * np.pi * np.array([0, 0, 1])

# Gamma = np.array([0, 0, 0])
# K = np.array([-0.375, 0.375, 0])
# W = np.array([-0.25, 0.5, 0.25])
# X = np.array([0, 0.5, 0.5])
# L = np.array([1, 1, 1])/2
# U = np.array([0.625, 0.625, 0.25])
# W1 = np.array([0.75, 0.5, 0.25])
# X1 = np.array([0.5,0.5, 0])

# Gamma = np.array([0, 0, 0])
# K = np.array([3/8,3/4,3/8])
# W = np.array([1/4,3/4,1/2])
# X = np.array([0, 0.5, 0.5])
# L = np.array([1, 1, 1])/2
# U = np.array([1/4,5/8,5/8])
# W1 = np.array([1/4,3/4,1/2])
# X1 = np.array([0, 0.5, 0.5])

stepN = magnitude_bi(U, W1)/graphres
# print(np.einsum('i,ik->k',W,BasisBZA),np.einsum('i,ik->k',K,BasisBZA),np.einsum('i,ik->k',X,BasisBZA) )
@nb.njit
def repcoord(a, b, c):
    return a*b1+b*b2+c*b3




z = np.array([np.array([1,1,1])/np.sqrt(3), np.array([1,-1,-1])/np.sqrt(3), np.array([-1,1,-1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)])
x = np.array([[-2,1,1],[-2,-1,-1],[2,1,-1], [2,-1,1]])/np.sqrt(6)
e = np.array([e0,e1,e2,e3])
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
                M[i, j] = 0
    return M

def gx(q):
    M = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if not np.dot(q,q) == 0:
                M[i,j] = np.dot(x[i], x[j]) - np.dot(x[i],q) * np.dot(x[j],q)/ np.dot(q,q)
            else:
                M[i, j] = 0
    return M

def gTransverse(q):
    Transverse = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if not np.dot(q,q) == 0:
                Transverse[i,j] = - np.dot(z[i],q) * np.dot(z[j],q)/ np.dot(q,q)
            else:
                Transverse[i, j] = 0
    M = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            M[i, j] = np.dot(z[i], z[j])
    return M, Transverse


def gNSF(v):
    M = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            M[i,j] = contract('a,a, b, b->',z[i], v, z[j], v)
    return M

NN = -np.array([np.array([1 / 4, 1 / 4, 1 / 4]), np.array([1 / 4, -1 / 4, -1 / 4]), np.array([-1 / 4, 1 / 4, -1 / 4]), np.array([-1 / 4, -1 / 4, 1 / 4])])


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
    a1_temp = np.copy(a1, order='C')
    a2_temp = np.copy(a2, order='C')
    a1_rows = a1_temp.view([('', np.float64)] * a1_temp.shape[1])
    a2_rows = a2_temp.view([('', np.float64)] * a2_temp.shape[1])
    return np.array(np.where(np.in1d(a1_rows, a2_rows)==True)[0], dtype=int)

def indextoignore_tol(a1, a2, tol):
    dex = np.array([],dtype =int)
    for i in a2:
        temp = np.linalg.norm(a1 - i, axis=1)
        tempdex = np.array(np.where(temp <= tol)[0], dtype=int)
        dex = np.concatenate((dex, tempdex))
    return dex


# @nb.njit
def genBZ(d, m=1):
    dj = d*1j
    b = np.mgrid[0:m:dj, 0:m:dj, 0:m:dj].reshape(3,-1).T
    b = np.concatenate((b,genALLSymPointsBare()))
    return b


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

def FFphase_setup(Flux):
    dim = int(2*np.pi/np.abs(Flux))
    pi3unitcellCoord = np.zeros((dim**2,3))

    for i in range(dim):
        for j in range(dim):
            pi3unitcellCoord[dim*i+j] = [0, i, j]

    pi3unitcellCoord_r2_shifted = np.mod(pi3unitcellCoord + [0, 1, 0], dim)
    pi3unitcellCoord_r3_shifted = np.mod(pi3unitcellCoord + [0, 0, 1], dim)

    p3unitcell_r2 = np.zeros((dim**2,dim**2))
    p3unitcell_r3 = np.zeros((dim**2,dim**2))

    for i in range(dim**2):
        for j in range(dim**2):
            p3unitcell_r2[i,j] = (pi3unitcellCoord_r2_shifted[j] == pi3unitcellCoord[i]).all()
            p3unitcell_r3[i,j] = (pi3unitcellCoord_r3_shifted[j] == pi3unitcellCoord[i]).all()

    pi3unitcell = np.array([
        np.eye(dim**2),
        np.eye(dim**2),
        p3unitcell_r2,
        p3unitcell_r3
    ])


    A_pi_p3 = np.zeros((dim**2,4))
    A_pi_p3[:,0] = 0
    A_pi_p3[:,3] = 0
    for i in range(dim**2):
        A_pi_p3[i, 1] = -Flux*(pi3unitcellCoord[i,1]-pi3unitcellCoord[i,2])
        A_pi_p3[i, 2] = -Flux*pi3unitcellCoord[i,2]

    A_pi_rs_traced_p3 = np.zeros((dim**2,4,4))
    for i in range(dim**2):
        for j in range(4):
            for k in range(4):
                A_pi_rs_traced_p3[i,j,k] = np.real(np.exp(1j * (A_pi_p3[i,j] - A_pi_p3[i,k])))

    A_pi_rs_traced_pp_p3 = np.zeros((dim**2,4,4))


    for i in range(dim**2):
        for j in range(4):
            for k in range(4):
                A_pi_rs_traced_pp_p3[i,j,k] = np.real(np.exp(1j * (A_pi_p3[i,j] + A_pi_p3[i,k])))


    A_pi_rs_rsp_p3 = np.zeros((dim**2,dim**2,4,4))

    for i in range(dim**2):
        for j in range(dim**2):
            for k in range(4):
                for l in range(4):
                    A_pi_rs_rsp_p3[i,j,k,l] = np.real(np.exp(1j * (A_pi_p3[i,k] - A_pi_p3[j,l])))


    A_pi_rs_rsp_pp_p3 = np.zeros((dim**2,dim**2,4,4))

    for i in range(dim**2):
        for j in range(dim**2):
            for k in range(4):
                for l in range(4):
                    A_pi_rs_rsp_pp_p3[i,j,k,l] = np.real(np.exp(1j * (A_pi_p3[i,k] + A_pi_p3[j,l])))
    return pi3unitcell, A_pi_p3, pi3unitcellCoord, A_pi_rs_traced_p3, A_pi_rs_traced_pp_p3, A_pi_rs_rsp_p3, A_pi_rs_rsp_pp_p3


piunitcellCoord = np.array([[0,0,0],[0,1,0],[0,0,1],[0,1,1]])



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
    N = magnitude_bi(A, B)
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

# Gamma = np.array([0, 0, 0])
# X = np.array([0, 0.5, 0.5])
# K = np.array([-0.375, 0.375, 0])
# W = np.array([-0.25, 0.5, 0.25])
#
#
# L = np.array([1, 1, 1])/2
# U = np.array([0.625, 0.625, 25])
# W1 = np.array([0.75, 0.5, 0.25])
# X1 = np.array([0.5,0.5, 0])

#Path to 1-10
GammaX = drawLine(Gamma, X, stepN)
XW = drawLine(X, W, stepN)
WK = drawLine(W, K, stepN)
KGamma = drawLine(K, Gamma, stepN)

#Path to 111 and then 001
GammaL = drawLine(Gamma, L, stepN)
LU = drawLine(L, U, stepN)
UW1 = drawLine(U, W1, stepN)
W1X1 = drawLine(W1, X1, stepN)
X1Gamma = drawLine(X1, Gamma, stepN)

gGamma1 = 0
gX = magnitude_bi(Gamma, X)
gW = gX + magnitude_bi(X, W)
gK = gW + magnitude_bi(W, K)

gGamma2 = gK + magnitude_bi(K, Gamma)
gL = gGamma2 + magnitude_bi(Gamma, L)
gU = gL + magnitude_bi(L, U)
gW1 = gU + magnitude_bi(U, W1)
gX1 = gW1 + magnitude_bi(W1, X1)
gGamma3 = gX1 + magnitude_bi(X1, Gamma)

graphGammaX = np.linspace(gGamma1, gX, len(GammaX))
graphXW = np.linspace(gX, gW, len(XW))
graphWK = np.linspace(gW, gK, len(WK))
graphKGamma = np.linspace(gK, gGamma2, len(KGamma))
graphGammaL = np.linspace(gGamma2, gL, len(GammaL))
graphLU = np.linspace(gL, gU, len(LU))
graphUW1 = np.linspace(gU, gW1, len(UW1))
graphW1X1 = np.linspace(gW1, gX1, len(W1X1))
graphX1Gamma = np.linspace(gX1, gGamma3, len(X1Gamma))

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
GammaALL = np.array([0, 0, 0])
KALL = np.array([[0.375, 0.375, 0],
                 [0.375, 0, 0.375],
                 [0, 0.375, 0.375],
                 [-0.375, 0.375, 0],
                 [-0.375, 0, 0.375],
                 [0, -0.375, 0.375],
                 [0.375, -0.375, 0],
                 [0.375, 0, -0.375],
                 [0, 0.375, -0.375],
                 [-0.375, -0.375, 0],
                 [-0.375, 0, -0.375],
                 [0, -0.375, -0.375]])
XALL = np.array([[0, 0.5, 0.5],
                 [0.5, 0, 0.5],
                 [0.5, 0.5, 0],
                 [0, -0.5, 0.5],
                 [-0.5, 0, 0.5],
                 [-0.5, 0.5, 0],
                 [0, 0.5, -0.5],
                 [0.5, 0, -0.5],
                 [0.5, -0.5, 0],
                 [0, -0.5, -0.5],
                 [-0.5, 0, -0.5],
                 [-0.5, -0.5, 0]])
WALL = np.array([[0.25, 0.25, 0.5],
                 [0.25, 0.5, 0.25],
                 [0.5, 0.25, 0.25],
                 [-0.25, 0.25, 0.5],
                 [-0.25, 0.5, 0.25],
                 [-0.5, 0.25, 0.25],
                 [0.25, -0.25, 0.5],
                 [0.25, -0.5, 0.25],
                 [0.5, -0.25, 0.25],
                 [0.25, 0.25, -0.5],
                 [0.25, 0.5, -0.25],
                 [0.5, 0.25, -0.25],
                 [-0.25, -0.25, 0.5],
                 [-0.25, -0.5, 0.25],
                 [-0.5, -0.25, 0.25],
                 [0.25, -0.25, -0.5],
                 [0.25, -0.5, -0.25],
                 [0.5, -0.25, -0.25],
                 [-0.25, 0.25, -0.5],
                 [-0.25, 0.5, -0.25],
                 [-0.5, 0.25, -0.25],
                 [-0.25, -0.25, -0.5],
                 [-0.25, -0.5, -0.25],
                 [-0.5, -0.25, -0.25]])
LALL = np.array([[1, 1, 1],
                 [-1, 1, 1],
                 [1, -1, 1],
                 [1, 1, -1],
                 [-1, -1, 1],
                 [-1, 1, -1],
                 [1, -1, -1],
                 [-1, -1, -1]])/2
UALL = np.array([[0.625, 0.625, 0.25],
                 [0.625, 0.25, 0.625],
                 [0.25, 0.625, 0.625],
                 [-0.625, 0.625, 0.25],
                 [-0.625, 0.25, 0.625],
                 [-0.25, 0.625, 0.625],
                 [0.625, -0.625, 0.25],
                 [0.625, -0.25, 0.625],
                 [0.25, -0.625, 0.625],
                 [0.625, 0.625, -0.25],
                 [0.625, 0.25, -0.625],
                 [0.25, 0.625, -0.625],
                 [-0.625, -0.625, 0.25],
                 [-0.625, -0.25, 0.625],
                 [-0.25, -0.625, 0.625],
                 [-0.625, 0.625, -0.25],
                 [-0.625, 0.25, -0.625],
                 [-0.25, 0.625, -0.625],
                 [0.625, -0.625, -0.25],
                 [0.625, -0.25, -0.625],
                 [0.25, -0.625, -0.625],
                 [-0.625, -0.625, -0.25],
                 [-0.625, -0.25, -0.625],
                 [-0.25, -0.625, -0.625]])

@nb.njit
def equi_class_111(K1, K2):
    if (K1 == K2).all() or (K1 == np.array([K2[2], K2[0], K2[1]])).all() or (K1 == np.array([K2[1], K2[2], K2[0]])).all()\
            or (K1 == np.array([K2[1], K2[0], K2[2]])).all() or (K1 == np.array([K2[0], K2[2], K2[1]])).all() or (K1 == np.array([K2[2], K2[1], K2[0]])).all():
        return True
    else:
        return False
def gen_equi_class_111(K1):
    A1 = K1[:, [2, 0, 1]]
    A2 = K1[:, [1, 2, 0]]
    A3 = K1[:, [1, 0, 2]]
    A4 = K1[:, [0, 2, 1]]
    A5 = K1[:, [2, 1, 0]]
    return np.unique(np.concatenate((K1,A1,A2,A3,A4,A5)),axis=0)
@nb.njit
def equi_class_110(K1, K2):
    if (K1 == K2).all() or (K1 == np.array([K2[1], K2[0], K2[2]])).all():
        return True
    else:
        return False
def gen_equi_class_110(K1):
    A1 = K1[:, [1, 0, 2]]
    return np.unique(np.concatenate((K1,A1)),axis=0)

@nb.njit
def equi_class_100(K1, K2):
    if (K1 == K2).all() or (K1 == np.array([K2[1], K2[0], K2[2]])).all() \
            or (K1 == np.mod(np.array([K2[0], K2[1], -K2[0]-K2[1]-K2[2]]),1)).all()\
            or (K1 == np.mod(np.array([K2[1], K2[0], -K2[0]-K2[1]-K2[2]]),1)).all():
        return True
    else:
        return False
def gen_equi_class_100(K1):
    temp1 = np.zeros(K1.shape)
    temp2 = np.zeros(K1.shape)
    temp1[:,0] = K1[:,0]
    temp2[:,1] = K1[:,1]
    A1 = K1[:, [1, 0, 2]]
    A2 = np.mod(K1 - temp1 - temp2,1)
    A3 = np.mod(K1[:, [1, 0, 2]] - temp1 - temp2,1)
    return np.unique(np.concatenate((K1,A1,A2,A3)),axis=0)

@nb.njit
def equi_class_0_flux(K1, K2):
    if (K1 == K2).all() or (K1==np.mod(K2+np.array([0,0,0.5]),1)).all() \
            or (K1 == np.mod(K2 + np.array([0, 0.5, 0]),1)).all() or (K1==K2+np.mod(np.array([0,0.5,0.5]),1)).all():
        return True
    else:
        return False
def gen_equi_class_0_flux(K1):
    A1 = K1 + np.array([0,0,0.5])
    A2 = K1 + np.array([0,0.5,0])
    A3 = K1 + np.array([0,0.5,0.5])
    return np.unique(np.concatenate((K1,A1,A2,A3)),axis=0)
@nb.njit
def equi_class_pi_flux(K1, K2):
    return False
def gen_equi_class_pi_flux(K1):
    return K1

@nb.njit
def equi_class_pp00_flux(K1, K2):
    if (K1 == K2).all() or (K1==np.mod(K2+np.array([0,0,0.5]),1)).all():
        return True
    else:
        return False
def gen_equi_class_pp00_flux(K1):
    A1 = K1 + np.array([0,0,0.5])
    return np.unique(np.concatenate((K1,A1)),axis=0)
@nb.njit
def equi_class_00pp_flux(K1, K2):
    if (K1 == K2).all() or (K1==np.mod(K2+np.array([0,0.5,0]),1)).all():
        return True
    else:
        return False
def gen_equi_class_00pp_flux(K1):
    A1 = K1 + np.array([0,0.5,0])
    return np.unique(np.concatenate((K1,A1)),axis=0)
@nb.njit
def equi_class_0pp0_flux(K1, K2):
    if (K1 == K2).all() or (K1==np.mod(K2+np.array([0,0.5,0]),1)).all():
        return True
    else:
        return False
def gen_equi_class_0pp0_flux(K1):
    A1 = K1 + np.array([0,0.5,0])
    return np.unique(np.concatenate((K1,A1)),axis=0)
@nb.njit
def equi_class_p00p_flux(K1, K2):
    if (K1 == K2).all() or (K1==np.mod(K2+np.array([0,0.5,0.5]),1)).all():
        return True
    else:
        return False
def gen_equi_class_p00p_flux(K1):
    A1 = K1 + np.array([0,0.5,0.5])
    return np.unique(np.concatenate((K1,A1)),axis=0)


@nb.njit('uint32(int32)')
def hash_32bit_4k(value):
    return (np.uint32(value) * np.uint32(27_644_437)) & np.uint32(0x0FFF)

@nb.njit(['int32[:](int32[:], int32[:])', 'int32[:](int32[::1], int32[::1])'])
def setdiff1d_nb_faster(arr1, arr2):
    out = np.empty_like(arr1)
    bloomFilter = np.zeros(4096, dtype=np.uint8)
    for j in range(arr2.size):
        bloomFilter[hash_32bit_4k(arr2[j])] = True
    cur = 0
    for i in range(arr1.size):
        # If the bloom-filter value is true, we know arr1[i] is not in arr2.
        # Otherwise, there is maybe a false positive (conflict) and we need to check to be sure.
        if bloomFilter[hash_32bit_4k(arr1[i])] and arr1[i] in arr2:
            continue
        out[cur] = arr1[i]
        cur += 1
    return out[:cur]
@nb.njit
def symmetry_equivalence(K, equi_relation):
    equiv_classes = []

    # Step 2: Find equivalence classes
    for i in range(K.shape[0]):
        x = K[i]
        added = False
        for equiv_class in equiv_classes:
            if equi_relation(x, equiv_class[0]):
                equiv_class.append(x)
                added = True
                break
        if not added:
            equiv_classes.append([x])
    representatives = np.zeros((len(equiv_classes),3),dtype=np.float64)
    for i in range(len(equiv_classes)):
        for j in range(3):
            representatives[i,j]=equiv_classes[i][0][j]

    return representatives


def determineEquivalence(n, flux):
    if (n == h110).all():
        A_pi_here, n1, n2 = constructA_pi_110(flux)
    elif (n == h111).all():
        A_pi_here, n1, n2 = constructA_pi_111(flux)
    elif (n == h001).all():
        A_pi_here, n1, n2 = constructA_pi_001(flux)
    return A_pi_here, n1, n2

def genALLSymPoints():
    d = 9 * 1j
    b = np.mgrid[-2*np.pi:2*np.pi:d, -2*np.pi:2*np.pi:d, -2*np.pi:2*np.pi:d].reshape(3, -1).T
    return b
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
        return np.pi*np.array([1,0,0])
    elif mu == 1:
        return np.pi*np.array([0,1,0])
    elif mu == 2:
        return np.pi*np.array([0,0,1])



def hnhltoK(H, L, K=0):
    return np.einsum('ij,k->ijk',H, np.array([-0.5,0.5,0])) \
        + np.einsum('ij,k->ijk',L, np.array([0.5,0.5,0])) \
        + K*np.array([0.5,0.5,1])

def hhltoK(H, L, K=0):
    return np.einsum('ij,k->ijk',H, np.array([0.5,0.5,1])) \
        + np.einsum('ij,k->ijk',L, np.array([0.5,0.5,0])) \
        + K*np.array([-0.5,0.5,0])
def hkztoK(H, K, L=0):
    return np.einsum('ij,k->ijk',H, np.array([0,0.5,0.5])) \
        + np.einsum('ij,k->ijk',K, np.array([0.5,0,0.5])) \
        + L*np.array([0.5, 0.5, 0])
def hhknktoK(H, K, L=0):
    return np.einsum('ij,k->ijk',H, np.array([0.5,0.5,1])) \
        + np.einsum('ij,k->ijk',K, np.array([-0.5,0.5,0])) \
        + L*np.array([0.5,0.5,0])

def hnhkkn2ktoK(H, K, L=0):
    return np.einsum('ij,k->ijk',H, np.array([-0.5,0.5,0])) \
        + np.einsum('ij,k->ijk',K, np.array([-0.5,-0.5,1])) \
        + L*np.array([0.5,0.5,0])


def hknkL(H,K,L):
    hk = hhknktoK(H, K).reshape((len(H) * len(K), 3))
    return np.einsum('ik,l->lik', hk, np.ones(len(L))) \
        + np.einsum('i,l,k->lik',np.ones(len(H)*len(K)),L, np.array([0.5,0.5,0]))

def hk0L(H,K,L):
    hk = hkztoK(H, K).reshape((len(H)*len(K), 3))
    return np.einsum('ik,l->lik', hk, np.ones(len(L))) \
        + np.einsum('i,l,k->lik',np.ones(len(H)*len(K)),L, np.array([0.5,0.5,0]))

def hhlK(H,K,L):
    hk = hnhltoK(H, L).reshape((len(H) * len(L), 3))
    return np.einsum('ik,l->lik', hk, np.ones(len(K))) \
        + np.einsum('i,l,k->lik',np.ones(len(H)*len(L)), K, np.array([-0.5,0.5,0]))



def q_scaplane(K):
    tempB = np.array([np.array([-1,1,1]),np.array([1,-1,1]),np.array([1,1,-1])])
    temp = np.einsum('ij,jk->ik', K, tempB)
    return temp


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

deltamin= 10
minLamK = 2

def gauss_quadrature_1D_pts(a, b, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    gauss_pts = (b - a) / 2 * nodes + (b + a) / 2
    weights *= (b - a) / 2
    return gauss_pts, weights


def gauss_quadrature_3D_pts(a, b, c, d, e, g, n):
    nodes1, weights1 = np.polynomial.legendre.leggauss(n)
    nodes2, weights2 = np.polynomial.legendre.leggauss(n)
    nodes3, weights3 = np.polynomial.legendre.leggauss(n)

    amp = np.array([(b-a)/2,(d-c)/2,(g-e)/2])
    # Map nodes from the interval [-1, 1] to the interval [a, b]
    gauss_pts = contract('k, ik->ik',amp,np.array(np.meshgrid(nodes1, nodes2, nodes3)).reshape(3, -1).T) + np.array([(a+b)/2,(c+d)/2,(e+g)/2])
    weights = contract('i,j,k->ijk', weights1, weights2, weights3).ravel()
    weights *= 0.125 * (b - a) * (d - c) * (g - e)
    return gauss_pts, weights
def integrate(f, gauss_pts, weights, *args):
    integral_approximation = np.dot(weights, f(gauss_pts, *args))
    return integral_approximation

def integrate_fixed(f, weights, *args):
    integral_approximation = np.dot(weights, f(*args))
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

def trapezoidal_rule_1d_pts(a, b, n):
    xi = np.linspace(a, b, n + 1)

    dx = (b - a) / n

    weight_x = np.full_like(xi, 1)
    weight_x[0] = weight_x[-1] = 1/2

    weights = weight_x*dx
    return xi, weights

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

def HanYan_GS(Jpm, Jzz, h ,n ,flux):
    Jpm = -2*Jpm
    fluxN = np.array([z[3],z[0],z[1],z[2]])
    g = 3*Jpm**3/(2*Jzz**2) + 5*Jpm**2/(4*Jzz**2)*(contract('k,ik->i', h*n, fluxN))**2 + 5*Jpm**2/(4*Jzz**2)*(contract('k,ik->i', h*n, fluxN))**2
    return np.dot(g, np.cos(flux))/4

def HanYan_g(Jpm, Jzz, h ,n):
    # Jpm = -2*Jpm
    fluxN = np.array([z[3],z[0],z[1],z[2]])
    g = 3*Jpm**3/(2*Jzz**2) + 5*Jpm**2/(4*Jzz**2)*(contract('k,ik->i', h*n, fluxN))**2 + 5*Jpm**2/(4*Jzz**2)*(contract('k,ik->i', h*n, fluxN))**2
    return g

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    print(array[idx])
    return idx

def gen_gauge_configurations(A_pi_here):
    size = A_pi_here.shape[0]
    A_pi_rs_traced_here = np.zeros((size, 4, 4), dtype=np.complex128)
    for i in range(size):
        for j in range(4):
            for k in range(4):
                A_pi_rs_traced_here[i, j, k] = np.exp(1j * (A_pi_here[i, j] - A_pi_here[i, k]))
    A_pi_rs_traced_pp_here = np.zeros((size, 4, 4), dtype=np.complex128)
    for i in range(size):
        for j in range(4):
            for k in range(4):
                A_pi_rs_traced_pp_here[i, j, k] = np.exp(1j * (A_pi_here[i, j] + A_pi_here[i, k]))
    A_pi_rs_rsp_here = np.zeros((size, size, 4, 4), dtype=np.complex128)
    for i in range(size):
        for j in range(size):
            for k in range(4):
                for l in range(4):
                    A_pi_rs_rsp_here[i, j, k, l] = np.exp(1j * (A_pi_here[i, k] - A_pi_here[j, l]))
    A_pi_rs_rsp_pp_here = np.zeros((size, size, 4, 4), dtype=np.complex128)
    for i in range(size):
        for j in range(size):
            for k in range(4):
                for l in range(4):
                    A_pi_rs_rsp_pp_here[i, j, k, l] = np.exp(1j * (A_pi_here[i, k] + A_pi_here[j, l]))
    return A_pi_rs_traced_here, A_pi_rs_traced_pp_here, A_pi_rs_rsp_here, A_pi_rs_rsp_pp_here

def xi_mean_field_110(xi, n1, n2, n4, unitcellCoord):
    size = len(unitcellCoord)
    xitemp = np.zeros((size, 4),dtype=complex)
    for i in range(size):
        xitemp[i] = np.array([xi[0,0], xi[0,1]*np.exp(1j*np.pi*(n1*unitcellCoord[i,1]+n2*unitcellCoord[i,2]))\
                                 , xi[0,1]*np.exp(1j*np.pi*(n2*unitcellCoord[i,2]+n4)), xi[0,3]])
    return xitemp
def xi_mean_field_111(xi, n1, n5, unitcellCoord):
    size = len(unitcellCoord)
    xitemp = np.zeros((size, 4),dtype=complex)
    for i in range(size):
        xitemp[i] = np.array([xi[0,0], xi[0,1]*np.exp(1j*np.pi*n1*(unitcellCoord[i,1]*unitcellCoord[i,2]))\
                                 , xi[0,1]*np.exp(1j*np.pi*(n1*unitcellCoord[i,2]+n5)), xi[0,0]])
    return xitemp
def xi_mean_field_001(xi, n1, n2, n4, unitcellCoord):
    size = len(unitcellCoord)
    xitemp = np.zeros((size, 4),dtype=complex)
    for i in range(size):
        xitemp[i] = np.array([xi[0,0], xi[0,1]*np.exp(1j*np.pi*n1*(unitcellCoord[i,1]+unitcellCoord[i,2]))\
                                 , xi[0,1]*np.exp(1j*np.pi*(n2*unitcellCoord[i,2]+n4)), xi[0,0]])
    return xitemp
def chi_mean_field_110(chi, n1, n2, n3, n4, unitcellCoord):
    size = len(unitcellCoord)
    chitemp = np.zeros((size, 4, 4),dtype=complex)
    for i in range(size):
        chi01 = chi[0,1]*np.exp(1j*np.pi*(n1*unitcellCoord[i,1]+n2*unitcellCoord[i,2]))
        chi02 = chi[0,1]*np.exp(1j*np.pi*(n2*unitcellCoord[i,2]+n4))
        chi03 = chi[0,3]
        chi12 = chi[1,2] * np.exp(1j*np.pi*n1*unitcellCoord[i,1])
        chi13 = chi[1, 3]*np.exp(1j*np.pi*(n1*unitcellCoord[i,1]+n2*unitcellCoord[i,2]))
        chi23 = chi[1, 3]*np.exp(1j*np.pi*(n2*unitcellCoord[i,2]+n3+n4))
        A = np.array([[chi[0,0], chi01, chi02, chi03],
                               [chi01, chi[0,0], chi12, chi13],
                               [chi01, chi12, chi[0,0], chi23],
                               [chi01, chi13, chi23,chi[0,0]]])
        chitemp[i] = A
    return chitemp
def chi_mean_field_111(chi, n1, n5, unitcellCoord):
    size = len(unitcellCoord)
    chitemp = np.zeros((size, 4, 4),dtype=complex)
    for i in range(size):
        chi01 = chi[0,1]*np.exp(1j*np.pi*n1*(unitcellCoord[i,1]+unitcellCoord[i,2]))
        chi02 = chi[0,1]*np.exp(1j*np.pi*(n1*unitcellCoord[i,2]+n5))
        chi03 = chi[0,1]*np.exp(1j*np.pi*n5)
        chi12 = chi[1,2] * np.exp(1j*np.pi*n1*unitcellCoord[i,1])
        chi13 = chi[1, 2]*np.exp(1j*np.pi*n1*(unitcellCoord[i,1]+unitcellCoord[i,2]))
        chi23 = chi[1, 2]*np.exp(1j*np.pi*(n1*unitcellCoord[i,2]+n5))
        chitemp[i] = np.array([[0, chi01, chi02, chi03],
                               [chi01, 0, chi12, chi13],
                               [chi01, chi12, 0, chi23],
                               [chi01, chi13, chi23,0]])
    return chitemp
def chi_mean_field_001(chi, n1, n2, n4, n5, unitcellCoord):
    size = len(unitcellCoord)
    chitemp = np.zeros((size, 4, 4),dtype=complex)
    for i in range(size):
        chi01 = chi[0,1]*np.exp(1j*np.pi*n1*(unitcellCoord[i,1]+unitcellCoord[i,2]))
        chi02 = chi[0,1]*np.exp(1j*np.pi*(n2*unitcellCoord[i,2]+n4))
        chi03 = chi[0,3]
        chi12 = chi[1,2] * np.exp(1j*np.pi*n1*unitcellCoord[i,1])
        chi13 = chi[0, 1]*np.exp(1j*np.pi*(n1*unitcellCoord[i,1]+n2*unitcellCoord[i,2]+n4+n5))
        chi23 = chi[0, 1]*np.exp(1j*np.pi*(n2*unitcellCoord[i,2]+n4))
        chitemp[i] = np.array([[0, chi01, chi02, chi03],
                               [chi01, 0, chi12, chi13],
                               [chi01, chi12, 0, chi23],
                               [chi01, chi13, chi23,0]])
    return chitemp
def xi_mean_field(n, xi, n1, n2, n4, n5, unitcellCoord):
    xi[0,0] = np.real(xi[0,0])
    if (n==h110).all():
        return xi_mean_field_110(xi, n1, n2, n4, unitcellCoord)
    elif (n==h111).all():
        return xi_mean_field_111(xi, n1, n5, unitcellCoord)
    else:
        return xi_mean_field_001(xi, n1, n2, n4, unitcellCoord)
def chi_mean_field(n, chi, n1, n2, n3, n4, n5, unitcellCoord):
    if (n==h110).all():
        return chi_mean_field_110(chi, n1, n2, n3, n4, unitcellCoord)
    elif (n==h111).all():
        return chi_mean_field_111(chi, n1, n5, unitcellCoord)
    else:
        return chi_mean_field_001(chi, n1, n2, n4, n5, unitcellCoord)
asize = 100
def plthhlfBZ(ax):
    Gamma = np.array([[0,0]])
    K = np.array([[3/4,0]])
    L = np.array([[0.5,0.5]])
    U = np.array([[1/4,1]])
    X = np.array([[0,1]])
    ax.scatter(Gamma[:,0], Gamma[:,1], zorder=10,s=asize)
    ax.scatter(K[:,0], K[:,1], zorder=10,s=asize)
    ax.scatter(U[:,0], U[:,1], zorder=10,s=asize)
    ax.scatter(L[:,0], L[:,1], zorder=10,s=asize)
    ax.scatter(X[:,0], X[:,1], zorder=10,s=asize)
    ax.text(Gamma[:,0]+0.05, Gamma[:,1]+0.05, r'$\Gamma$', zorder=10, color="w")
    ax.text(K[:,0]+0.05, K[:,1]+0.05, r'$K$', zorder=10, color="w")
    ax.text(U[:,0]+0.05, U[:,1]+0.05, r'$U$', zorder=10, color="w")
    ax.text(L[:,0]+0.05, L[:,1]+0.05, r'$L$', zorder=10, color="w")
    ax.text(X[:,0], X[:,1]+0.05, r'$X$', zorder=10, color="w")
    BZ = np.array([[3/4,0],
                   [1/4,1],
                   [-1/4,1],
                   [-3/4,0],
                   [-1/4,-1],
                   [1/4,-1],
                   [3/4,0]])
    ax.plot(BZ[:,0], BZ[:,1], '--', zorder=9, color='w')
def plthk0fBZ(ax):
    Gamma = np.array([[0,0]])
    K = np.array([[3/4,3/4]])
    W = np.array([[1/2, 1]])
    X = np.array([[0,1]])
    ax.scatter(Gamma[:,0], Gamma[:,1], zorder=10,s=asize)
    ax.scatter(K[:,0], K[:,1], zorder=10,s=asize)
    ax.scatter(W[:,0], W[:,1], zorder=10,s=asize)
    ax.scatter(X[:,0], X[:,1], zorder=10,s=asize)
    ax.text(Gamma[:,0]+0.05, Gamma[:,1]+0.05, r'$\Gamma$', zorder=10, color="w")
    ax.text(K[:,0]+0.05, K[:,1]+0.05, r'$K$', zorder=10, color="w")
    ax.text(W[:,0]+0.05, W[:,1]+0.05, r'$W$', zorder=10, color="w")
    ax.text(X[:,0]+0.05, X[:,1]+0.05, r'$X$', zorder=10, color="w")
    BZ = np.array([[1,0.5],
                   [0.5,1],
                   [-0.5,1],
                   [-1,0.5],
                   [-1,-0.5],
                   [-0.5,-1],
                   [0.5,-1],
                   [1,-0.5],
                   [1,0.5]])
    ax.plot(BZ[:,0], BZ[:,1], '--', zorder=9, color='w')

def plthh2kfBZ(ax):
    return 0
    # Gamma = np.array([[0,0]])
    # K = np.array([[3/4,0]])
    # X = np.array([[0.5,0.5]])
    # ax.scatter(Gamma[:,0], Gamma[:,1], zorder=10,s=asize)
    # ax.scatter(K[:,0], K[:,1], zorder=10)
    # # ax.scatter(W[:,0], W[:,1], zorder=10)
    # ax.scatter(X[:,0], X[:,1], zorder=10)
    # # ax.scatter(U[:,0], U[:,1], zorder=10)
    # # ax.scatter(L[:,0], L[:,1], zorder=10)
    # ax.text(Gamma[:,0]+0.05, Gamma[:,1]+0.05, r'$\Gamma$')
    # ax.text(K[:,0]+0.05, K[:,1]+0.05, r'$K$')
    # ax.text(X[:,0]+0.05, X[:,1]+0.05, r'$X$')
    # BZ = np.array([[3/4,0],
    #                [1/3,1/3],
    #                [-1/3,1/3],
    #                [-3/4,0],
    #                [-1/3,-1/3],
    #                [1/3,-1/3],
    #                [3/4,0]])
    #
    # ax.plot(BZ[:,0], BZ[:,1], '--', zorder=9, color='w')

def plt1dhhlfBZ(ax):
    Gamma = np.array([[0,0],[2,0],[-2,0]])
    K = np.array([[1,0],[-1,0]])
    ax.scatter(Gamma[:,0], Gamma[:,1], zorder=10,s=100)
    ax.scatter(K[:,0], K[:,1], zorder=10,s=100)
    ax.text(np.array([0.05]), np.array([0.05]), r'$\Gamma_\beta$', zorder=10, color="w")
    ax.text(np.array([1.05]), np.array([0.05]), r'$K_\beta$', zorder=10, color="w")
    ax.axvline(x=1, color='w', label='axvline - full height', linestyle='dashed')
    ax.axvline(x=-1, color='w', label='axvline - full height', linestyle='dashed')

def XYZparambuilder(JPm, JPmax, JP1m, JP1max, nK):
    JH = np.mgrid[JPm:JPmax:1j * nK, JP1m:JP1max:1j * nK].reshape(2,-1).T.reshape(nK,nK,2)
    split = int(nK/2)
    JH1 = JH[0:split, :,:]
    JH2 = np.transpose(JH[split:, split:, :], (1,0,2))
    if nK % 2 == 0:
        for i in range(1,split):
            JH2[i-1,i:] = JH1[i-1,i-1:split-1]
        temp = np.zeros((split,2*split+1,2))
        temp[:,0:split] = JH2
        temp[:, split:] = JH1[:, split-1:,:]
        return temp.reshape((split*(2*split+1),2))


def inverseXYZparambuilder(M):
    split = len(M)
    temp = np.copy(M[0:split, 0:split])
    temp = temp.T
    orig = np.zeros((2*split,2*split))
    orig[0:split,:] = M[:,1:]
    orig[split:,split:]=temp

    for i in range(2*split):
        for j in range(2*split):
            if i > j:
                orig[i,j] = orig[j,i]
    return orig


# JH = np.arange(100).reshape((10,10))
# nK = 10
# split = int(nK/2)
# JH1 = JH[0:split, :]
# JH2 = np.transpose(JH[split:, split:], (1,0))
#
# for i in range(1,split):
#     JH2[i-1,i:] = JH1[i-1,i-1:split-1]
# A = np.zeros((split,2*split+1))
# A[:,0:split] = JH2
# A[:, split:] = JH1[:, split-1:]
#
# A = A.reshape((1,-1)).T
#
# A = A.reshape((5,11))
#
# B = inverseXYZparambuilder(A)
# # A = XYZparambuilder(0, 1, 10)
# print()