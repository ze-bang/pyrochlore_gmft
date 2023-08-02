import numpy as np
import matplotlib.pyplot as plt
import warnings
from numba import jit



def ifFBZ(k):
    b1, b2, b3 = k
    if np.any(abs(k) > 2 * np.pi):
        return False
    elif abs(b1 + b2 + b3) < 3 * np.pi and abs(b1 - b2 + b3) < 3 * np.pi and abs(-b1 + b2 + b3) < 3 * np.pi and abs(
            b1 + b2 - b3) < 3 * np.pi:
        return True
    else:
        return False



def genBZ( d):
    d = d * 1j
    b = np.mgrid[-2 * np.pi:2 * np.pi:d, -2 * np.pi:2 * np.pi:d, -2 * np.pi:2 * np.pi:d].reshape(3, -1).T
    BZ = []
    for x in b:
        if ifFBZ(x):
            BZ += [x]
    return BZ

def z(mu):
    if mu == 0:
        return -np.array([1,1,1])/np.sqrt(3)
    if mu == 1:
        return np.array([-1,1,1])/np.sqrt(3)
    if mu == 2:
        return np.array([1,-1,1])/np.sqrt(3)
    if mu == 3:
        return np.array([1,1,-1])/np.sqrt(3)

class zeroFluxSolver:

    def __init__(self, Jpm, h=0, n=np.array([0,0,0]), eta=1, kappa=2, lam=2, BZres=20, graphres=20, omega=0, Jzz=1):
        self.Jzz = Jzz
        self.Jpm = Jpm
        self.kappa = kappa
        self.eta = [eta, 1]
        self.h = h
        self.n = n
        self.omega = omega

        self.tol = 1e-3
        self.lams = np.array([lam, lam], dtype=float)

        self.b0 = np.pi * np.array([1, 1, 1])
        self.b1 = np.pi * np.array([-1, 1, 1])
        self.b2 = np.pi * np.array([1, -1, 1])
        self.b3 = np.pi * np.array([1, 1, -1])

        self.e0 = np.array([0, 0, 0])/2
        self.e1 = np.array([0, 1, 1])/2
        self.e2 = np.array([1, 0, 1])/2
        self.e3 = np.array([1, 1, 0])/2

        self.minLams = np.zeros(2)
        self.maglam = lam

        self.Gamma = np.array([0, 0, 0])
        self.L = np.pi * np.array([1, 1, 1])
        self.X = 2*np.pi * np.array([0, 1, 0])
        self.W = 2*np.pi * np.array([0, 1, 1 / 2])
        self.K = 2*np.pi * np.array([0, 3 / 4, 3 / 4])
        self.U = 2*np.pi * np.array([1 / 4, 1, 1 / 4])

        self.BZres = BZres
        self.graphres = graphres
        self.bigB = genBZ(BZres)

        self.M = np.zeros((2,len(self.bigB)))
        self.MF = []
        self.E =[]

        self.GammaX = self.drawLine(self.Gamma, self.X, graphres)
        self.XW = self.drawLine(self.X, self.W, graphres)
        self.WK = self.drawLine(self.W, self.K, graphres)
        self.KGamma = self.drawLine(self.K, self.Gamma, graphres)
        self.GammaL = self.drawLine(self.Gamma, self.L, graphres)
        self.LU = self.drawLine(self.L, self.U, graphres)
        self.UW = self.drawLine(self.U,self.W, graphres)

        self.bigK = np.concatenate((self.GammaX, self.XW, self.WK, self.KGamma, self.GammaL, self.LU, self.UW))

        self.dGammaX = np.zeros((2,graphres))
        self.dXW = np.zeros((2,graphres))
        self.dWK = np.zeros((2,graphres))
        self.dKGamma = np.zeros((2,graphres))
        self.dGammaL = np.zeros((2,graphres))
        self.dLU = np.zeros((2,graphres))
        self.dUW = np.zeros((2,graphres))

        self.V = np.zeros((len(self.bigB),2,2))
        self.Vt = np.zeros((len(self.bigB),2,2))
        self.didit = False
        self.Mset = False

#pyrochlore brillouin zone

#basis vector


    #Critcal Points

    def repcoord(self, a, b, c):
        return a*self.b1+b*self.b2+c*self.b3

    def realcoord(self, r):
        r1, r2, r3 = r
        return r1*self.e1 +r2*self.e2 + r3* self.e3




    #alpha = 1 for A = -1 for B


    def NNtest(self,mu):
        if mu == 0:
            return self.e0
        if mu == 1:
            return self.e1
        if mu == 2:
            return self.e2
        if mu == 3:
            return self.e3

    def step(self,mu):
        if mu == 0:
            return np.array([0,0,0])
        if mu == 1:
            return np.array([1,0,0])
        if mu == 2:
            return np.array([0,1,0])
        if mu == 3:
            return np.array([0,0,1])

    def NN(self, mu):
        if mu == 0:
            return np.array([-1/4, -1/4, -1/4])
        if mu == 1:
            return np.array([-1/4, 1/4, 1/4])
        if mu == 2:
            return np.array([1/4, -1/4, 1/4])
        if mu == 3:
            return np.array([1/4, 1/4, -1/4])


    def neta(self, alpha):
        if alpha == 0:
            return 1
        elif alpha == 1:
            return -1


    def exponent_zero(self, k, alpha, mu, nu):
        return np.exp(-1j*self.neta(alpha)*(np.dot(k,self.NNtest(mu)-self.NNtest(nu))))


    def exponent_mag(self, k, alpha):
        temp =0
        for mu in range(4):
            temp += - 1/2 * self.h * np.dot(self.n, z(mu)) * np.cos(np.dot(k, self.neta(alpha)*self.NN(mu)))
        return temp


    def M_zero(self, k, alpha):
        temp = 0
        for i in range(4):
            for j in range(4):
                if not i==j:
                    temp += self.exponent_zero(k, alpha, i, j)
        temp = -self.Jpm*self.eta[alpha]/4 * temp


        return np.real(temp)


    def M_tot(self, k):
        M = np.zeros((len(k), 2,2), dtype=complex)
        M[:, 0, 0] = self.M_zero(k, 0)
        M[:, 1, 1] = self.M_zero(k, 1)
        M[:, 0, 1] = self.exponent_mag(k, 0)
        M[:, 1, 0] = self.exponent_mag(k, 1)
        E, V = np.linalg.eigh(M)
        if self.Mset == False:
            self.MF = M
            self.Mset = True
        # print(V)
        # print(E)
        return np.real(E)

    def M_single(self, k):
        M = np.zeros((2,2), dtype=complex)
        M[0, 0] = self.M_zero(k, 0)
        M[1, 1] = self.M_zero(k, 1)
        M[0, 1] = self.exponent_mag(k, 0)
        M[1, 0] = self.exponent_mag(k, 1)
        E, V = np.linalg.eig(M)
        # print(V)
        return np.real(E)

    # def Green(self, lams):
    #     temp = self.MF + np.diag(lams+self.omega/(2*self.Jzz))
    #     E,V = np.linalg.eig(temp)
    #     return np.real(E)
    #
    # def rho_true(self, alpha, lams):
    #     E = self.Green(lams)
    #     return np.mean(self.Jzz/np.sqrt(2*self.Jzz*E), axis=0)[alpha, alpha]


    def minLam(self):
        # k = obliqueProj(k)
        temp = np.amin(self.M.T, axis=0)
        # temp = self.M_single(self.L)

        self.minLams = -temp
        return 0

    def phase_test(self):
        try:
            rho = np.array([self.rho_zero(0, self.minLams), self.rho_zero(1, self.minLams)])
            if (rho < self.kappa).any():
                return 1
            else:
                return 0
        except:
            return 1

    def setM(self):
        self.M = self.M_tot(self.bigB).T
        return 1

    def setupALL(self):
        self.setM()
        self.minLam()
        return 0


    def E_zero_fixed(self, alpha, lams):
        val = np.sqrt(2 * self.Jzz * self.eta[1-alpha] * (lams[alpha] + self.M[alpha]))
        return np.real(val)


    def E_zero(self, k, alpha, lams):
        val = np.sqrt(2 * self.Jzz * self.eta[1-alpha] * (lams[alpha] + self.M_tot(k)[:, alpha]))
        return np.real(val)


    def rho_zero(self, alpha, lams):
        temp = np.mean(self.Jzz*self.eta[alpha] /self.E_zero_fixed(alpha, lams))
        return temp
    def findLambda_zero(self,alpha):
        warnings.filterwarnings("error")
        lamMin = 0
        lamMax = 100
        rhoguess = 10
        # print(self.kappa)
        while(np.absolute(rhoguess-self.kappa) >= self.tol):
             self.lams[alpha] = (lamMin+lamMax)/2
             try:
                 rhoguess = self.rho_zero(alpha, self.lams)
                 # rhoguess = self.rho_zero(alpha, self.lams)
                 if rhoguess - self.kappa > 0:
                     lamMin = self.lams[alpha]
                 else:
                     lamMax = self.lams[alpha]
             except Exception as e:
                 # print(e)
                 lamMin = self.lams[alpha]
             # print([lamMin, lamMax, rhoguess])
             # if lamMax < self.minLams[alpha]:
             #     self.lams[alpha] = -1000
             #     return 1
             if lamMax == 0:
                 self.lams[alpha] = -1000
                 return 1
        return 0


    #graphing BZ

    def drawLine(self, A, B, N):
        return np.linspace(A, B, N)

    def dispersion_zero(self, P, alpha, lams):
        temp = self.E_zero(P, alpha, lams)
        return temp

    def findLambda(self):
        self.findLambda_zero(0)
        self.findLambda_zero(1)
        return 1


    def calDispersionA(self, alpha):
        self.dGammaX[alpha] = self.dispersion_zero(self.GammaX, alpha, self.lams)
        self.dXW[alpha] = self.dispersion_zero(self.XW, alpha, self.lams)
        self.dWK[alpha] = self.dispersion_zero(self.WK, alpha, self.lams)
        self.dKGamma[alpha] = self.dispersion_zero(self.KGamma, alpha, self.lams)
        self.dGammaL[alpha] = self.dispersion_zero(self.GammaL, alpha, self.lams)
        self.dLU[alpha] = self.dispersion_zero(self.LU, alpha, self.lams)
        self.dUW[alpha] = self.dispersion_zero(self.UW, alpha, self.lams)

    def calDispersion(self):
        self.calDispersionA(0)
        self.calDispersionA(1)

        self.didit = True



    def graph(self, alpha, show):
        if not self.didit:
            self.calDispersion()
        plt.plot(np.linspace(-0.5, 0, self.graphres), self.dGammaX[alpha], 'b')
        plt.plot(np.linspace(0, 0.3, self.graphres), self.dXW[alpha] , 'b')
        plt.plot(np.linspace(0.3, 0.5, self.graphres), self.dWK[alpha], 'b')
        plt.plot(np.linspace(0.5, 0.9, self.graphres), self.dKGamma[alpha], 'b')
        plt.plot(np.linspace(0.9, 1.3, self.graphres), self.dGammaL[alpha], 'b')
        plt.plot(np.linspace(1.3, 1.6, self.graphres), self.dLU[alpha], 'b')
        plt.plot(np.linspace(1.6, 1.85, self.graphres),self.dUW[alpha], 'b')
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
        if show:
            plt.show()

    def graph_even(self, alpha, show):
        if not self.didit:
            self.calDispersion()
        plt.plot(np.linspace(-0.5, 0, self.graphres), self.dGammaX[alpha], 'b')
        plt.plot(np.linspace(0, 0.5, self.graphres), self.dXW[alpha] , 'b')
        plt.plot(np.linspace(0.5, 1, self.graphres), self.dWK[alpha], 'b')
        plt.plot(np.linspace(1, 1.5, self.graphres), self.dKGamma[alpha], 'b')
        plt.plot(np.linspace(1.5, 2, self.graphres), self.dGammaL[alpha], 'b')
        plt.plot(np.linspace(2, 2.5, self.graphres), self.dLU[alpha], 'b')
        plt.plot(np.linspace(2.5, 3, self.graphres),self.dUW[alpha], 'b')
        if show:
            plt.show()

    def gap(self, alpha):
        if not self.didit:
            self.calDispersion()
        return self.dGammaX[alpha][0]


    def GS(self, alpha):
        if not self.didit:
            self.calDispersion()
        return np.mean(self.E_zero_fixed(alpha, self.lams)) - self.lams[alpha]


    def EMAX(self, alpha):
        if not self.didit:
            self.calDispersion()
        return max(self.dGammaX[alpha])



