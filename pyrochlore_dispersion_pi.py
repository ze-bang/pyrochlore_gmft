import numpy as np
import matplotlib.pyplot as plt
import warnings
from multiprocessing import Pool

def neta(alpha):
    if alpha == 0:
        return 1
    else:
        return -1

def ifFBZ(k):
    b1, b2, b3 = k
    if np.any(abs(k) > 2*np.pi):
        return False
    elif abs(b1+b2+b3)<3*np.pi and abs(b1-b2+b3)<3*np.pi and abs(-b1+b2+b3)<3*np.pi and abs(b1+b2-b3)<3*np.pi:
        return True
    else:
        return False

def genBZ(d):
    d = d*1j
    b = np.mgrid[-2*np.pi:2*np.pi:d, -2*np.pi:2*np.pi:d, -2*np.pi:2*np.pi:d].reshape(3,-1).T
    BZ = []
    for x in b:
        if ifFBZ(x):
            BZ += [x]
    return BZ

def step(mu):
    if mu == 0:
        return np.array([0,0,0])
    if mu == 1:
        return np.array([1,0,0])
    if mu == 2:
        return np.array([0,1,0])
    if mu == 3:
        return np.array([0,0,1])






def drawLine(A, B, N):
    return np.linspace(A, B, N)




def mod2pi(a):
    while a>= 2*np.pi:
        a = a - 2*np.pi
    return a

def A_pi(r1,r2):
    bond = r1-r2
    r1, r2, r3 = r1

    if np.all(bond == step(0)):
        return 0
    if np.all(bond == step(1)):
        return mod2pi(np.pi*(r2+r3))
    if np.all(bond == step(2)):
        return mod2pi(np.pi*r3)
    if np.all(bond == step(3)):
        return 0
    if np.all(bond == -step(1)):
        return mod2pi(np.pi*(r2+r3))
    if np.all(bond == -step(2)):
        return mod2pi(np.pi*r3)
    if np.all(bond == -step(3)):
        return 0

def findS(r):
    for i in range (4):
        if np.all(r == unitCell(i)):
            return i
    return -1

def unitCell(mu):
    if mu == 0:
        return np.array([0,0,0])
    if mu == 1:
        return np.array([0,1,0])
    if mu == 2:
        return np.array([0,0,1])
    if mu == 3:
        return np.array([0,1,1])

def NNtest(mu):
    if mu == 0:
        return np.array([0, 0, 0])/2
    if mu == 1:
        return np.array([0, 1, 1])/2
    if mu == 2:
        return np.array([1, 0, 1])/2
    if mu == 3:
        return np.array([1, 1, 0])/2

def z(mu):
    if mu == 0:
        return -np.array([1,1,1])/np.sqrt(3)
    if mu == 1:
        return np.array([-1,1,1])/np.sqrt(3)
    if mu == 2:
        return np.array([1,-1,1])/np.sqrt(3)
    if mu == 3:
        return np.array([1,1,-1])/np.sqrt(3)
class piFluxSolver:

    def __init__(self, Jpm, h=0, n=np.array([0,0,0]), eta=1, kappa=2, lam=2, res=20, Jzz=1):
        self.Jzz = Jzz
        self.Jpm = Jpm
        self.kappa = kappa
        self.eta = [eta, 1]
        self.tol = 1e-3
        self.lams =[lam, lam]
        self.h = h
        self.n = n

        self.b0 = np.pi * np.array([1, 1, 1])
        self.b1 = np.pi * np.array([-1, 1, 1])
        self.b2 = np.pi * np.array([1, -1, 1])
        self.b3 = np.pi * np.array([1, 1, -1])
        #
        # self.e0 = np.array([0, 0, 0])/2
        # self.e1 = np.array([0, 1, 1])/2
        # self.e2 = np.array([1, 0, 1])/2
        # self.e3 = np.array([1, 1, 0])/2

        self.E = []
        self.V= []

        self.mindex = -1
        self.minLams = np.zeros(2)

        # self.e0 = np.array([0, 0, 0])
        # self.e1 = np.array([1, 0, 0])
        # self.e2 = np.array([0, 1, 0])
        # self.e3 = np.array([0, 0, 1])

        self.Gamma = np.array([0, 0, 0])

        self.L = np.pi * np.array([1, 1, 1])
        self.X = 2*np.pi * np.array([0, 1, 0])
        self.W = 2*np.pi * np.array([0, 1, 1 / 2])
        self.K = 2*np.pi * np.array([0, 3 / 4, 3 / 4])
        self.U = 2*np.pi * np.array([1 / 4, 1, 1 / 4])

        self.res =res
        self.bigB = genBZ(res)
        self.M = np.zeros((2, len(self.bigB), 4))

        self.GammaX = drawLine(self.Gamma, self.X, res)
        self.XW = drawLine(self.X, self.W, res)
        self.WK = drawLine(self.W, self.K, res)
        self.KGamma = drawLine(self.K, self.Gamma, res)
        self.GammaL = drawLine(self.Gamma, self.L, res)
        self.LU = drawLine(self.L, self.U, res)
        self.UW = drawLine(self.U,self.W, res)

        self.dGammaX =  np.zeros((8,res))
        self.dXW =  np.zeros((8,res))
        self.dWK =  np.zeros((8,res))
        self.dKGamma =  np.zeros((8,res))
        self.dGammaL =  np.zeros((8,res))
        self.dLU =  np.zeros((8,res))
        self.dUW = np.zeros((8,res))

        self.didit=False
        self.updated=True

    #alpha = 1 for A = -1 for B




    def b(self, mu):
        if mu == 0:
            return self.b1
        if mu == 1:
            return self.b2
        if mu == 2:
            return self.b3

    def obliqueProj(self,W):
        M = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                M[i][j] = np.dot(self.b(i), self.b(j))

        y = np.zeros((3, 1))
        for i in range(3):
            y[i] = np.dot(W, self.b(i))
        return np.array(np.matmul(np.linalg.inv(M), y)).T[0]

    def exponent_pi(self, k, alpha, mu, nu, rs1, rs2):
        rs = rs1 - neta(alpha) * step(mu)

        f = np.real(np.exp(1j * neta(alpha) * (A_pi(rs, rs2) - A_pi(rs, rs1))))

        return f * np.exp(1j * neta(alpha) * (np.dot(k, NNtest(nu) - NNtest(mu))))


    def M_pi_term(self, k, alpha, rs1, rs2, mu, nu):
        temp = -self.Jpm*self.eta[alpha]/4 * self.exponent_pi(k, alpha, mu, nu, rs1, rs2)
        return temp

    def M_pi_mag_term(self, k, alpha, rs1, mu):
        rs = rs1 - neta(alpha) * step(mu)
        temp = 1 / 2 * self.h * neta(alpha) * np.dot(self.n, z(mu)) * np.exp(1j * A_pi(rs, rs1)) * np.exp(1j * np.dot(k, neta(alpha) * NNtest(mu)))
        return temp

    def M_pi_sub(self, k, rs, alpha):
        M = np.zeros((4, 4), dtype=complex)
        for i in range(4):
            for j in range(4):
                if not i==j:
                    mu = unitCell(rs) + neta(alpha)*step(i)
                    nu = unitCell(rs) + neta(alpha)*step(j)
                    rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                    rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                    index1 = findS(rs1)
                    index2 = findS(rs2)
                    M[index1][index2] += self.M_pi_term(k, alpha, mu, nu, i, j)
                    M[i][index2] += -self.M_pi_mag_term(k, alpha, rs2, j)
                    M[index2][i] += -np.conj(self.M_pi_mag_term(k, 1-alpha, rs2, j))

        return M

    def M_pi(self, k, alpha):
        bigM = np.zeros((4,4,4), dtype=complex)
        for i in range(4):
            bigM[i] = self.M_pi_sub(k,i,alpha)
        M = np.sum(bigM, axis=0)
        E, V = np.linalg.eigh(M)
        self.V = V
        return E



    def setM(self):
        for i in range(len(self.bigB)):
            self.M[0][i] = self.M_pi(self.bigB[i], 0)
            self.M[1][i] = self.M_pi(self.bigB[i], 1)
        return 1
    #
    #
    # def setM(self):
    #     with Pool() as pool:
    #         result = pool.map(self.setM_serial,range(len(self.bigB)))
    #     return 1


    def E_pi_fixed(self, alpha, lams):
        # k = obliqueProj(k)
        temp = self.M[alpha]
        try:
            val = np.sqrt(2 * self.Jzz * self.eta[1-alpha] * (lams[alpha] + temp))
            return val
        except:
            return 0


    def E_pi(self,k, alpha, lams):
        # k = obliqueProj(k)
        temp = self.M_pi(k,alpha)
        val = np.sqrt(2 * self.Jzz * self.eta[1-alpha] * (lams[alpha] + temp))
        return val


    def setmindex(self):
        self.calAllDispersion()
        mins = 1000
        mindex = -1
        for i in range(len(self.dLU[0])):
            if self.dLU[0][i] < mins:
                mins = self.dLU[0][i]
                mindex = i
        self.mindex = mindex

        return 0

    def minLam(self, alpha):
        # k = obliqueProj(k)
        self.setmindex()
        temp = self.M_pi(self.LU[self.mindex],alpha)[0]
        if temp == 0:
            temp = -1000
        self.minLams[alpha] = - temp
        return 0

    def setupALL(self):
        self.setM()
        self.minLam(0)
        self.minLam(1)
        return 0
    def setE(self, lams):
        self.E = np.zeros((2, len(self.bigB), 4))
        self.E[0] = self.E_pi_fixed(0, lams)
        self.E[1] = self.E_pi_fixed(1, lams)

    def rho_pi(self, alpha, lams):
        if self.updated:
            self.setE(lams)
        tempc = self.Jzz*self.eta[alpha]/self.E[alpha]
        temp = np.mean(tempc)
        return temp


    def findLambda_pi(self,alpha):
        warnings.filterwarnings("error")
        lamMin = 0
        lamMax = 100
        rhoguess = 10
        while(np.absolute(rhoguess-self.kappa) >= self.tol):
             self.lams[alpha] = (lamMin+lamMax)/2

             try:
                 rhoguess = self.rho_pi(alpha, self.lams)

                 if rhoguess - self.kappa > 0:
                     lamMin = self.lams[alpha]
                 else:
                     lamMax = self.lams[alpha]
             except:
                 lamMin = self.lams[alpha]

             # if lamMax < self.minLams[alpha]:
             #     self.lams[alpha] = -1000
             #     return 1
             # print([self.lams[alpha],rhoguess])
             if lamMax == 0:
                 self.lams[alpha] = -1000
                 return 1
        return 0

    #graphing BZ

    def dispersion_pi(self, P, alpha, lams):
        temp = []
        for i in P:
            temp += [self.E_pi(i, alpha, lams)]
        return np.array(temp)

    def findLambda(self):
        self.findLambda_pi(0)
        self.findLambda_pi(1)
        return 1

    def calDispersion(self, alpha):
        indexS = alpha*4
        indexE = (alpha+1)*4
        # print(self.lams)
        self.dGammaX[indexS:indexE] = self.dispersion_pi(self.GammaX, alpha, self.lams).T
        self.dXW[indexS:indexE] = self.dispersion_pi(self.XW, alpha, self.lams).T
        self.dWK[indexS:indexE] = self.dispersion_pi(self.WK, alpha, self.lams).T
        self.dKGamma[indexS:indexE] = self.dispersion_pi(self.KGamma, alpha, self.lams).T
        self.dGammaL[indexS:indexE] = self.dispersion_pi(self.GammaL, alpha, self.lams).T
        self.dLU[indexS:indexE] = self.dispersion_pi(self.LU, alpha, self.lams).T
        self.dUW[indexS:indexE] = self.dispersion_pi(self.UW, alpha, self.lams).T

    def calAllDispersion(self):
        self.calDispersion(0)
        self.calDispersion(1)

    def graphbranch(self, index, color):
        plt.plot(np.linspace(-0.5, 0, self.res), self.dGammaX[index], color)
        plt.plot(np.linspace(0, 0.3, self.res), self.dXW[index], color)
        plt.plot(np.linspace(0.3, 0.5, self.res), self.dWK[index], color)
        plt.plot(np.linspace(0.5, 1, self.res), self.dKGamma[index], color)
        plt.plot(np.linspace(1, 1.4, self.res), self.dGammaL[index], color)
        plt.plot(np.linspace(1.4, 1.7, self.res), self.dLU[index], color)
        plt.plot(np.linspace(1.7, 1.85, self.res), self.dUW[index], color)
    def graph(self, alpha, show):
        if not self.didit:
            self.calAllDispersion()

        for i in range(4):
            index = alpha*4 + i
            self.graphbranch(index, 'b')

        # self.graphbranch(0, 'b')
        # self.graphbranch(1, 'r')
        # self.graphbranch(2, 'y')
        # self.graphbranch(3, 'g')

        plt.ylabel(r'$\omega/J_{zz}$')
        plt.axvline(x=-0.5, color='b', label='axvline - full height', linestyle='dashed')
        plt.axvline(x=0, color='b', label='axvline - full height', linestyle='dashed')
        plt.axvline(x=0.3, color='b', label='axvline - full height', linestyle='dashed')
        plt.axvline(x=0.5, color='b', label='axvline - full height', linestyle='dashed')
        plt.axvline(x=1, color='b', label='axvline - full height', linestyle='dashed')
        plt.axvline(x=1.4, color='b', label='axvline - full height', linestyle='dashed')
        plt.axvline(x=1.7, color='b', label='axvline - full height', linestyle='dashed')
        plt.axvline(x=1.85, color='b', label='axvline - full height', linestyle='dashed')
        xlabpos = [-0.5,0,0.3,0.5,1,1.4,1.7,1.85]
        labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
        plt.xticks(xlabpos, labels)

        if show:
            plt.show()


    def gap(self, alpha):
        if not self.didit:
            self.calAllDispersion()
        mins = []
        for i in range(4):
            index = alpha*4 + i
            mins += [min(self.dLU[index])]

        return min(mins)

    def GS(self, alpha):
        if self.updated:
            self.setE(self.res, self.lams)

        temp = np.mean(self.E)- self.lams[alpha]
        return temp
    def EMAX(self, alpha):
        if not self.didit:
            self.calAllDispersion()
        return max(self.dLU[4*alpha+2])