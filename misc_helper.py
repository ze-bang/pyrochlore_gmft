import time

import numpy as np
import numba as nb
from opt_einsum import contract
import math
from flux_stuff import *
from scipy.optimize import minimize
# import pyrochlore_exclusive_boson as pyeb
# import matplotlib.pyplot as plt
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

# Gamma = np.array([0, 0, 0])
# L = np.pi * np.array([1, 1, 1])
# X = 2 * np.pi * np.array([1, 0, 0])
# W = 2 * np.pi * np.array([1, 0, 1 / 2])
# K = 2 * np.pi * np.array([3/4, 0, 3 / 4])
# U = 2 * np.pi * np.array([1, 1/4, 1 / 4])


Gamma = np.array([0, 0, 0])
L = np.array([1, 1, 1])/2
X = np.array([0, 0.5, 0.5])
W = np.array([0.25, 0.75, 0.5])
K = np.array([0.375, 0.75, 0.375])
U = np.array([0.25, 0.625, 0.625])



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
            M[i,j] = contract('a,a, b, b->',z[i], v, z[j], v)
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

graphGammaX = np.linspace(gGamma1, gX, len(GammaX))
graphXW = np.linspace(gX, gW1, len(XW))
graphWK = np.linspace(gW1, gK, len(WK))
graphKGamma = np.linspace(gK, gGamma2, len(KGamma))
graphGammaL = np.linspace(gGamma2, gL, len(GammaL))
graphLU = np.linspace(gL, gU, len(LU))
graphUW = np.linspace(gU, gW2, len(UW))

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
        A_pi_here = constructA_pi_110(flux)
        equi_class_field = equi_class_110
        gen_equi_class_field = gen_equi_class_110
        if (flux == np.zeros(4)).all():
            equi_class_flux = equi_class_0_flux
            gen_equi_class_flux = gen_equi_class_0_flux
        elif (flux == np.pi*np.ones(4)).all():
            equi_class_flux = equi_class_pi_flux
            gen_equi_class_flux = gen_equi_class_pi_flux
        elif (flux == np.array([np.pi,np.pi,0,0])).all():
            equi_class_flux = equi_class_pp00_flux
            gen_equi_class_flux = gen_equi_class_pp00_flux
        else:
            equi_class_flux = equi_class_00pp_flux
            gen_equi_class_flux = gen_equi_class_00pp_flux
    elif (n == h111).all():
        A_pi_here = constructA_pi_111(flux)
        equi_class_field = equi_class_111
        gen_equi_class_field = gen_equi_class_111
        if (flux == np.zeros(4)).all():
            equi_class_flux = equi_class_0_flux
            gen_equi_class_flux = gen_equi_class_0_flux
        else:
            equi_class_flux = equi_class_pi_flux
            gen_equi_class_flux = gen_equi_class_pi_flux
    elif (n == h001).all():
        A_pi_here = constructA_pi_001(flux)
        equi_class_field = equi_class_100
        gen_equi_class_field = gen_equi_class_100
        if (flux == np.zeros(4)).all():
            equi_class_flux = equi_class_0_flux
            gen_equi_class_flux = gen_equi_class_0_flux
        elif (flux == np.pi*np.ones(4)).all():
            equi_class_flux = equi_class_pi_flux
            gen_equi_class_flux = gen_equi_class_pi_flux
        elif (flux == np.array([np.pi,0,0,np.pi])).all():
            equi_class_flux = equi_class_p00p_flux
            gen_equi_class_flux = gen_equi_class_p00p_flux
        else:
            equi_class_flux = equi_class_0pp0_flux
            gen_equi_class_flux = gen_equi_class_0pp0_flux
    return A_pi_here, equi_class_field, equi_class_flux, gen_equi_class_field, gen_equi_class_flux

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
        return np.pi*np.array([1,0,0])
    elif mu == 1:
        return np.pi*np.array([0,1,0])
    elif mu == 2:
        return np.pi*np.array([0,0,1])



def hhltoK(H, L):
    return np.einsum('ij,k->ijk',H, np.array([0.5,0.5,1])) + np.einsum('ij,k->ijk',L, np.array([0.5,0.5,0]))
def hkztoK(H, L):
    return np.einsum('ij,k->ijk',H, np.array([0,0.5,0.5])) + np.einsum('ij,k->ijk',L, np.array([0.5,0,0.5]))
def hkktoK(H, K):
    return np.einsum('ij,k->ijk',H, np.array([0.5,0.5,1])) \
        + np.einsum('ij,k->ijk',K, np.array([-0.5,0.5,0])) \

def hknkL(H,K,L):
    hk = hkktoK(H, K).reshape((len(H)*len(K), 3))
    return np.einsum('ik,l->ilk', hk, np.ones(len(L))) \
        + np.einsum('i,l,k->ilk',np.ones(len(H)*len(K)),L, np.array([0.5,0.5,0]))


def hhlscaplane(H,L):
    return np.einsum('ij,k->ijk',H, np.array([1,1,0]))
def hk0scaplane(H,L):
    return np.einsum('ij,k->ijk',H, np.array([1,0,0]))+ np.einsum('ij,k->ijk',L, np.array([0,1,0]))
def hkkscaplane(H,L):
    return np.einsum('ij,k->ijk',H, np.array([1,1,0]))+ np.einsum('ij,k->ijk',L, np.array([1,-1,0]))

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


# def HanYan_g(Jpm, Jzz, h, n):
#     # Jpm = -2*Jpm
#     if (n==np.array([0,0,1])).all():
#         E = - (7*h**6+23*h**4*Jpm**2+147*h**2*Jpm*4)/(48*Jzz**5) \
#             + (35*h**4*Jpm-108*h**2*Jpm**3-351*Jpm**5)/(72*Jzz**4) \
#             - 5*h**2*Jpm**2/(12*Jzz**3) - 3*Jpm**3/(2*Jzz**2)
#         E = E * np.ones(4)
#         return E
#     elif (n==np.array([1,1,1])/np.sqrt(3)).all():
#         E = np.zeros(4)
#         E[1] = - (7*h**6+1293*h**4*Jpm**2+13095*h**2*Jpm*4)/(1296*Jzz**5) \
#             - (35*h**4*Jpm+468*h**2*Jpm**3+1053*Jpm**5)/(216*Jzz**4) + \
#             - 5*h**2*Jpm**2/(4*Jzz**3) - 3*Jpm**3/(2*Jzz**2)
#         E[0] = E[2] = E[3] = - (21*h**6+31*h**4*Jpm**2+309*h**2*Jpm*4)/(432*Jzz**5) \
#             - (35*h**4*Jpm+276*h**2*Jpm**3+1053*Jpm**5)/(216*Jzz**4) + \
#             - 5*h**2*Jpm**2/(36*Jzz**3) - 3*Jpm**3/(2*Jzz**2)
#         return E
#     else:
#         E = np.zeros(4)
#         E[0] = E[1] = - (25*h**6+237*h**4*Jpm**2)/(36*Jzz**5) \
#             - (44*h**2*Jpm**3+117*Jpm**5)/(24*Jzz**4) + \
#             - 5*h**2*Jpm**2/(6*Jzz**3) - 3*Jpm**3/(2*Jzz**2)
#         E[2] = E[3] = -(11*h**2*Jpm*4)/(24*Jzz**5) \
#             - (28*h**2*Jpm**3+117*Jpm**5)/(24*Jzz**4) + \
#             - 3*Jpm**3/(2*Jzz**2)
#         return E
# def HanYan_GS(Jpm, Jzz, h, n, flux):
#     # Jpm = -2*Jpm
#     if (n==np.array([0,0,1])).all():
#         E = - (7*h**6+23*h**4*Jpm**2+147*h**2*Jpm*4)/(48*Jzz**5) \
#             + (35*h**4*Jpm-108*h**2*Jpm**3-351*Jpm**5)/(72*Jzz**4) \
#             - 5*h**2*Jpm**2/(12*Jzz**3) - 3*Jpm**3/(2*Jzz**2)
#         E = E * np.ones(4)
#         return -np.dot(E, np.cos(flux))/4
#     elif (n==np.array([1,1,1])/np.sqrt(3)).all():
#         E = np.zeros(4)
#         E[1] = - (7*h**6+1293*h**4*Jpm**2+13095*h**2*Jpm*4)/(1296*Jzz**5) \
#             - (35*h**4*Jpm+468*h**2*Jpm**3+1053*Jpm**5)/(216*Jzz**4) + \
#             - 5*h**2*Jpm**2/(4*Jzz**3) - 3*Jpm**3/(2*Jzz**2)
#         E[0] = E[2] = E[3] = - (21*h**6+31*h**4*Jpm**2+309*h**2*Jpm*4)/(432*Jzz**5) \
#             - (35*h**4*Jpm+276*h**2*Jpm**3+1053*Jpm**5)/(216*Jzz**4) + \
#             - 5*h**2*Jpm**2/(36*Jzz**3) - 3*Jpm**3/(2*Jzz**2)
#         return -np.dot(E, np.cos(flux))/4
#     else:
#         E = np.zeros(4)
#         E[0] = E[1] = - (25*h**6+237*h**4*Jpm**2)/(36*Jzz**5) \
#             - (44*h**2*Jpm**3+117*Jpm**5)/(24*Jzz**4) + \
#             - 5*h**2*Jpm**2/(6*Jzz**3) - 3*Jpm**3/(2*Jzz**2)
#         E[2] = E[3] = (11*h**2*Jpm*4)/(24*Jzz**5) \
#             - (28*h**2*Jpm**3+117*Jpm**5)/(24*Jzz**4) + \
#             - 3*Jpm**3/(2*Jzz**2)
#         return -np.dot(E, np.cos(flux))/4
