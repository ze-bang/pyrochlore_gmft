import numpy as np
import matplotlib.pyplot as plt
import warnings
from sympy.utilities.iterables import multiset_permutations
from itertools import permutations
from numba import njit, jit
from numba.experimental import jitclass
from numba import float32, int16, boolean, complex64
from numba import types, typed, typeof

b1 = 2 * np.pi * np.array([-1, 1, 1])
b2 = 2 * np.pi * np.array([1, -1, 1])
b3 = 2 * np.pi * np.array([1, 1, -1])

e0 = np.array([0.0, 0.0, 0.0])
e1 = np.array([0, 1, 1]) / 2
e2 = np.array([1, 0, 1]) / 2
e3 = np.array([1, 1, 0]) / 2




def ifFBZ(k):
    b1, b2, b3 = k
    if np.any(np.abs(k) > 2 * np.pi):
        return False
    elif np.abs(b1 + b2 + b3) < 3 * np.pi and np.abs(b1 - b2 + b3) < 3 * np.pi and np.abs(-b1 + b2 + b3) < 3 * np.pi and np.abs(
            b1 + b2 - b3) < 3 * np.pi:
        return True
    else:
        return False


def genBZ(d):
    d = d * 1j
    b = np.mgrid[-2 * np.pi:2 * np.pi:d, -2 * np.pi:2 * np.pi:d, -2 * np.pi:2 * np.pi:d].reshape(3, -1).T
    BZ = []
    # print(b)
    for x in b:
        if ifFBZ(x):
            BZ += [x]
    return BZ
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

def BasisBZ(mu):
    if mu == 0:
        return np.pi*np.array([-1,1,1])
    if mu == 1:
        return np.pi*np.array([1,-1,1])
    if mu == 2:
        return np.pi*np.array([1,1,-1])

# def genBZ( d):
#     d = d * 1j
#     b = np.mgrid[-1: 1:d, -1: 1:d, -1: 1:d].reshape(3, -1).T
#     BZ = np.zeros((len(b), 3), dtype=np.single)
#     for i in range (len(b)):
#         BZ[i] = b[i, 0] *BasisBZ(0) + b[i, 1] *BasisBZ(1) + b[i, 2] *BasisBZ(2)
#     return BZ

def z(mu):
    if mu == 0:
        return -np.array([1,1,1])/np.sqrt(3)
    if mu == 1:
        return np.array([-1,1,1])/np.sqrt(3)
    if mu == 2:
        return np.array([1,-1,1])/np.sqrt(3)
    if mu == 3:
        return np.array([1,1,-1])/np.sqrt(3)


def repcoord(a, b, c):
    return a*b1+b*b2+c*b3

def realcoord(r):
    r1, r2, r3 = r
    return r1*e1 +r2*e2 + r3* e3




#alpha = 1 for A = -1 for B

def NNtest(mu):
    if mu == 0:
        return e0
    if mu == 1:
        return e1
    if mu == 2:
        return e2
    if mu == 3:
        return e3

def step(mu):
    if mu == 0:
        return np.array([0,0,0])
    if mu == 1:
        return np.array([1,0,0])
    if mu == 2:
        return np.array([0,1,0])
    if mu == 3:
        return np.array([0,0,1])
def NN(mu):
    if mu == 0:
        return np.array([-1/4, -1/4, -1/4])
    if mu == 1:
        return np.array([-1/4, 1/4, 1/4])
    if mu == 2:
        return np.array([1/4, -1/4, 1/4])
    if mu == 3:
        return np.array([1/4, 1/4, -1/4])


def neta(alpha):
    if alpha == 0:
        return 1
    elif alpha == 1:
        return -1

formfactor = np.zeros((4,4,3))
for i in range(4):
    for j in range(4):
        formfactor[i,j,:] = NN(i)-NN(j)


NNN = np.array([[-1/4, -1/4, -1/4],[-1/4, 1/4, 1/4], [1/4, -1/4, 1/4], [1/4, 1/4, -1/4]])
ZZZ = np.array([[-1, -1, -1],[-1, 1, 1], [1, -1, 1], [1, 1, -1]])/np.sqrt(3)


def exponent_mag(h, n, k, alpha):
    temp = 0
    for i in range(4):
        temp = -1 * h * np.dot(n, z(i)) * np.cos(np.dot(k, neta(alpha)*NN(i)))
    return temp


def M_zero(Jpm, eta, k, alpha):
    temp = -Jpm/4 *eta[alpha]* np.exp(-1j*neta(alpha)*(np.einsum('ik, jlk->ijl',k, formfactor)))
    temp = np.einsum('ijk->i', temp)
    return temp

def M_zero_single(Jpm, eta, k, alpha):
    temp = -Jpm/4 *eta[alpha]* np.exp(-1j*neta(alpha)*(np.einsum('k, jlk->jl',k, formfactor)))
    temp = np.sum(temp)
    return temp

def M_true(k, Jpm, eta, h, n):
    M = np.zeros((len(k), 2,2), dtype=np.csingle)
    M[:, 0, 0] = M_zero(Jpm, eta, k, 0)
    M[:, 1, 1] = M_zero(Jpm, eta, k, 1)
    M[:, 0, 1] = exponent_mag(h, n, k, 0)
    M[:, 1, 0] = exponent_mag(h, n, k, 1)
    return M


def M_single(lams, k, Jzz, Jpm, eta, h, n):
    M = np.zeros((2,2), dtype=np.csingle)
    M[0, 0] = M_zero_single(Jpm, eta, k, 0)
    M[1, 1] = M_zero_single(Jpm, eta, k, 1)
    M[0, 1] = exponent_mag(h, n, k, 0)
    M[1, 0] = exponent_mag(h, n, k, 1)
    M = M + np.diag(lams)
    E, V = np.linalg.eig(M)
    return np.sqrt(2*Jzz*np.real(E))


def E_zero_true(lams, k, Jpm, eta, h, n):
    M = M_true(k, Jpm, eta, h, n)
    M = M + np.diag(lams)
    E, V = np.linalg.eigh(M)
    return [E,V]

def rho_true(Jzz, M, lams):
    temp = M + np.diag(lams)
    E,V = np.linalg.eigh(temp)
    Vt = np.real(np.einsum('ijk,ijk->ijk',V, np.conj(V)))
    Ep = np.mean(np.einsum('ijk, ik->ij', Vt, 1/np.sqrt(2*Jzz*E)), axis=0)
    return Ep


def findLambda_zero(M, Jzz, kappa, tol):
    warnings.filterwarnings("error")
    lamMin = np.zeros(2)
    lamMax = 10*np.ones(2)
    lams = (lamMin + lamMax) / 2
    rhoguess = rho_true(Jzz, M, lams)
    # print(self.kappa)
    for i in range(2):
        while np.absolute(rhoguess[i]-kappa) >= tol:
             lams[i] = (lamMin[i]+lamMax[i])/2
             # rhoguess = rho_true(Jzz, M, lams)
             try:
                 rhoguess = rho_true(Jzz, M, lams)
                 # rhoguess = self.rho_zero(alpha, self.lams)
                 if rhoguess[i] - kappa > 0:
                     lamMin[i] = lams[i]
                 else:
                     lamMax[i] = lams[i]
             except:
                 # print(e)
                 lamMin[i] = lams[i]
             print([lams[i], lamMin[i], lamMax[i], rhoguess[i]])
             # if lamMax == 0:
             #     break
    return lams


#graphing BZ

def drawLine(A, B, N):
    return np.linspace(A, B, N)

def dispersion_zero(lams, k, Jzz, Jpm, eta, h, n):
    temp = np.sqrt(2*Jzz*E_zero_true(lams, k, Jpm, eta, h, n)[0])
    return temp



Gamma = np.array([0, 0, 0])
L = np.pi * np.array([1, 1, 1])
X = 2*np.pi * np.array([0, 1, 0])
W = 2*np.pi * np.array([0, 1, 1 / 2])
K = 2*np.pi * np.array([0, 3 / 4, 3 / 4])
U = 2*np.pi * np.array([1 / 4, 1, 1 / 4])


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

# def populate(res):
#     temp = np.zeros((1,3))
#     for i in symK:
#         for j in symK:
#             if not (i == j).all():
#                 temp = np.concatenate((temp, np.linspace(i, j, res)))
#             else:
#                 temp = np.concatenate((temp, np.linspace(i, j, 1)))
#     return temp
#
# symK = populate(3)
graphres = 20

GammaX = drawLine(Gamma, X, graphres)
XW = drawLine(X, W, graphres)
WK = drawLine(W, K, graphres)
KGamma = drawLine(K, Gamma, graphres)
GammaL = drawLine(Gamma, L, graphres)
LU = drawLine(L, U, graphres)
UW = drawLine(U, W, graphres)

def calDispersion(lams, Jzz, Jpm, eta, h, n):

    dGammaX= dispersion_zero(lams, GammaX, Jzz, Jpm, eta, h, n)
    dXW= dispersion_zero(lams, XW, Jzz, Jpm, eta, h, n)
    dWK = dispersion_zero(lams, WK, Jzz, Jpm, eta, h, n)
    dKGamma = dispersion_zero(lams, KGamma, Jzz, Jpm, eta, h, n)
    dGammaL = dispersion_zero(lams, GammaL, Jzz, Jpm, eta, h, n)
    dLU= dispersion_zero(lams, LU, Jzz, Jpm, eta, h, n)
    dUW = dispersion_zero(lams, UW, Jzz, Jpm, eta, h, n)

    for i in range(2):
        plt.plot(np.linspace(-0.5, 0, graphres), dGammaX[:,i], 'b')
        plt.plot(np.linspace(0, 0.3, graphres), dXW[:, i] , 'b')
        plt.plot(np.linspace(0.3, 0.5, graphres), dWK[:, i], 'b')
        plt.plot(np.linspace(0.5, 0.9, graphres), dKGamma[:, i], 'b')
        plt.plot(np.linspace(0.9, 1.3, graphres), dGammaL[:, i], 'b')
        plt.plot(np.linspace(1.3, 1.6, graphres), dLU[:, i], 'b')
        plt.plot(np.linspace(1.6, 1.85, graphres),dUW[:, i], 'b')
    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=-0.5, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.3, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.5, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.9, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.3, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.6, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.85, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [-0.5,0,0.3,0.5,0.9,1.3,1.6,1.85]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)


def minCal(lams, q, Jzz, Jpm, eta, h, n, K):
    temp = np.zeros((len(q),2))
    mins = np.sqrt(2 * Jzz * E_zero_true(lams, K, Jpm, eta, h, n)[0])
    for i in range(len(q)):
        temp[i] = np.amin(np.sqrt(2 * Jzz * E_zero_true(lams, q[i]+K, Jpm, eta, h, n)[0]) + mins, axis=0)
    return temp

def maxCal(lams, q, Jzz, Jpm, eta, h, n, K):
    temp = np.zeros((len(q),2))
    mins = np.sqrt(2 * Jzz * E_zero_true(lams, K, Jpm, eta, h, n)[0])
    for i in range(len(q)):
        temp[i] = np.amax(np.sqrt(2 * Jzz * E_zero_true(lams, q[i]+K, Jpm, eta, h, n)[0]) + mins, axis=0)
    return temp

def loweredge(lams, Jzz, Jpm, eta, h, n, K):
    dGammaX= minCal(lams, GammaX, Jzz, Jpm, eta, h, n, K)
    dXW= minCal(lams, XW, Jzz, Jpm, eta, h, n, K)
    dWK = minCal(lams, WK, Jzz, Jpm, eta, h, n, K)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, eta, h, n, K)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, eta, h, n, K)
    dLU= minCal(lams, LU, Jzz, Jpm, eta, h, n, K)
    dUW = minCal(lams, UW, Jzz, Jpm, eta, h, n, K)

    for i in range(2):
        plt.plot(np.linspace(-0.5, 0, graphres), dGammaX[:,i], 'w')
        plt.plot(np.linspace(0, 0.3, graphres), dXW[:, i] , 'w')
        plt.plot(np.linspace(0.3, 0.5, graphres), dWK[:, i], 'w')
        plt.plot(np.linspace(0.5, 0.9, graphres), dKGamma[:, i], 'w')
        plt.plot(np.linspace(0.9, 1.3, graphres), dGammaL[:, i], 'w')
        plt.plot(np.linspace(1.3, 1.6, graphres), dLU[:, i], 'w')
        plt.plot(np.linspace(1.6, 1.85, graphres),dUW[:, i], 'w')
    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=-0.5, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.3, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.5, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.9, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.3, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.6, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.85, color='w', label='axvline - full height', linestyle='dashed')
    xlabpos = [-0.5,0,0.3,0.5,0.9,1.3,1.6,1.85]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)

def upperedge(lams, Jzz, Jpm, eta, h, n, K):
    dGammaX= maxCal(lams, GammaX, Jzz, Jpm, eta, h, n, K)
    dXW= maxCal(lams, XW, Jzz, Jpm, eta, h, n, K)
    dWK = maxCal(lams, WK, Jzz, Jpm, eta, h, n, K)
    dKGamma = maxCal(lams, KGamma, Jzz, Jpm, eta, h, n, K)
    dGammaL = maxCal(lams, GammaL, Jzz, Jpm, eta, h, n, K)
    dLU= maxCal(lams, LU, Jzz, Jpm, eta, h, n, K)
    dUW = maxCal(lams, UW, Jzz, Jpm, eta, h, n, K)

    for i in range(2):
        plt.plot(np.linspace(-0.5, 0, graphres), dGammaX[:,i], 'w')
        plt.plot(np.linspace(0, 0.3, graphres), dXW[:, i] , 'w')
        plt.plot(np.linspace(0.3, 0.5, graphres), dWK[:, i], 'w')
        plt.plot(np.linspace(0.5, 0.9, graphres), dKGamma[:, i], 'w')
        plt.plot(np.linspace(0.9, 1.3, graphres), dGammaL[:, i], 'w')
        plt.plot(np.linspace(1.3, 1.6, graphres), dLU[:, i], 'w')
        plt.plot(np.linspace(1.6, 1.85, graphres),dUW[:, i], 'w')
    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=-0.5, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.3, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.5, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=0.9, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.3, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.6, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=1.85, color='w', label='axvline - full height', linestyle='dashed')
    xlabpos = [-0.5,0,0.3,0.5,0.9,1.3,1.6,1.85]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)

def gap(M, lams):
    temp = M + np.diag(lams)
    E,V = np.linalg.eigh(temp)
    # E = np.sqrt(E)
    temp = np.amin(E)
    print("Gap is " + str(temp))
    return temp

def EMAX(M, lams):
    temp = M + np.diag(lams)
    E,V = np.linalg.eigh(temp)
    temp = np.amax(E)
    return temp

def GS(lams, k, Jzz, Jpm, eta, h, n):
    return np.mean(dispersion_zero(lams, k, Jzz, Jpm, eta, h, n), axis=0) - lams


class zeroFluxSolver:
    def __init__(self, Jpm, h=0, n=np.array([0,0,0]), eta=1, kappa=2, lam=2, BZres=20, graphres=20, omega=0, Jzz=1):
        self.Jzz = Jzz
        self.Jpm = Jpm
        self.kappa = kappa
        self.eta = np.array([eta, 1], dtype=np.single)
        self.h = h
        self.n = n
        self.omega = omega

        self.tol = 1e-3
        self.lams = np.array([lam, lam], dtype=np.single)


        # self.symK = self.genALLSymPoints()
        # self.symK = self.populate(BZres)

        self.BZres = BZres
        self.graphres = graphres
        self.bigB = np.concatenate((genBZ(BZres), symK))

        # print(self.bigB)


        self.E =np.zeros(len(self.bigB))

        self.MF = M_true(self.bigB, self.Jpm, self.eta, self.h, self.n)


    def findLambda(self):
        self.lams = findLambda_zero(self.MF, self.Jzz, self.kappa, self.tol)

    def M_true(self, k):
        return M_true(k, self.Jpm, self.eta, self.h, self.n)
    def E_zero(self, k):
        return E_zero_true(self.lams, k, self.Jpm, self.eta, self.h, self.n)
    def gap(self):
        return np.sqrt(2*self.Jzz*gap(self.MF, self.lams))

    def graph(self, show):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n)
        if show:
            plt.show()

    def E_single(self, k):
        return M_single(self.lams, k, self.Jzz, self.Jpm, self.eta, self.h, self.n)

    def EMAX(self):
        return np.sqrt(2*self.Jzz*EMAX(self.MF, self.lams))

    def graph_loweredge(self, show):
        loweredge(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB)
        if show:
            plt.show()

    def graph_upperedge(self, show):
        upperedge(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB)
        if show:
            plt.show()