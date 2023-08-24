import numpy as np
from itertools import permutations
import math
import numba as nb

graphres=50

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
X = 2 * np.pi * np.array([0, 1, 0])
W = 2 * np.pi * np.array([0, 1, 1 / 2])
K = 2 * np.pi * np.array([0, 3 / 4, 3 / 4])
U = 2 * np.pi * np.array([1 / 4, 1, 1 / 4])

stepN = magnitude(np.abs(U-W))/graphres


@nb.njit
def repcoord(a, b, c):
    return a*b1+b*b2+c*b3


@nb.njit
def realcoord(r):
    r1, r2, r3 = r
    return r1*e1 +r2*e2 + r3* e3

@nb.njit
def z(mu):
    if mu == 0:
        return -np.array([1,1,1])/np.sqrt(3)
    if mu == 1:
        return np.array([-1,1,1])/np.sqrt(3)
    if mu == 2:
        return np.array([1,-1,1])/np.sqrt(3)
    if mu == 3:
        return np.array([1,1,-1])/np.sqrt(3)

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

@nb.njit
def NN(mu):
    if mu == 0:
        return np.array([-1 / 4, -1 / 4, -1 / 4])
    if mu == 1:
        return np.array([-1 / 4, 1 / 4, 1 / 4])
    if mu == 2:
        return np.array([1 / 4, -1 / 4, 1 / 4])
    if mu == 3:
        return np.array([1 / 4, 1 / 4, -1 / 4])


@nb.njit
def ifFBZ(k):
    b1, b2, b3 = k
    if np.any(abs(k) > 2*np.pi):
        return False
    elif abs(b1+b2+b3)<3*np.pi and abs(b1-b2+b3)<3*np.pi and abs(-b1+b2+b3)<3*np.pi and abs(b1+b2-b3)<3*np.pi:
        return True
    else:
        return False


# def genBZ(d):
#     d = d*1j
#     b = np.mgrid[-2*np.pi:2*np.pi:d, -2*np.pi:2*np.pi:d, -2*np.pi:2*np.pi:d].reshape(3,-1).T
#     BZ = []
#     for x in b:
#         if ifFBZ(x):
#             BZ += [x]
#     return BZ



def genBZ(d):
    d = d*1j
    b = np.mgrid[0:1:d, 0:1:d, 0:1:d].reshape(3,-1).T
    basis = np.einsum('jl, ij->il',BasisBZA, b)
    return basis


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



@nb.njit
def A_pi(r1,r2):
    bond = r1-r2
    r1, r2, r3 = r1
    if np.all(bond == step(0)):
        return 0
    if np.all(bond == step(1)):
        return np.pi*(r2+r3) % (2*np.pi)
    if np.all(bond == step(2)):
        return np.pi*r3 % (2*np.pi)
    if np.all(bond == step(3)):
        return 0
    if np.all(bond == -step(1)):
        return np.pi*(r2+r3) % (2*np.pi)
    if np.all(bond == -step(2)):
        return np.pi*r3 % (2*np.pi)
    if np.all(bond == -step(3)):
        return 0


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
     [0,1,0,1]],
])

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
            M[i][j] = np.dot(b(i), b(j))

    y = np.zeros((3, 1))
    for i in range(3):
        y[i] = np.dot(W, b(i))
    return np.array(np.matmul(np.linalg.inv(M), y)).T[0]



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

def genALLSymPoints():
    pG = np.array(list(set(permutations(Gamma))))
    pL = np.array(list(set(permutations(L))))
    pX = np.array(list(set(permutations(X))))
    pW = np.array(list(set(permutations(W))))
    pK = np.array(list(set(permutations(K))))
    pU = np.array(list(set(permutations(U))))
    Lp = np.array(list(set(permutations(np.pi*np.array([-1,1,1])))))
    Wp = np.array(list(set(permutations(2*np.pi*np.array([0,-1,1/2])))))
    Kp = np.array(list(set(permutations(2*np.pi*np.array([0,-3/4,3/4])))))
    Up = np.array(list(set(permutations(2*np.pi*np.array([-1/4,1,1/4])))))
    Upp = np.array(list(set(permutations(2 * np.pi * np.array([1 / 4, -1, 1 / 4])))))
    A = np.concatenate((pG,pL,pX,pW,pK,pU))
    Ap = np.concatenate((Lp, Up, Upp))
    A = np.concatenate((A, -A, Ap, -Ap, Wp, Kp))
    return A

symK = genALLSymPoints()

# graphres = 20
#
# GammaX = drawLine(Gamma, X, graphres)
# XW = drawLine(X, W, graphres)
# WK = drawLine(W, K, graphres)
# KGamma = drawLine(K, Gamma, graphres)
# GammaL = drawLine(Gamma, L, graphres)
# LU = drawLine(L, U, graphres)
# UW = drawLine(U, W, graphres)