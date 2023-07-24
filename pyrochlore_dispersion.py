import numpy as np
import matplotlib.pyplot as plt
import warnings


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

    def __init__(self, Jpm, h=0, n=np.array([0,0,0]), eta=1, kappa=2, lam=2, res=20, Jzz=1):
        self.Jzz = Jzz
        self.Jpm = Jpm
        self.kappa = kappa
        self.eta = [eta, 1]
        self.h = h
        self.n = n

        self.tol = 1e-3
        self.lams = [lam, lam]

        self.b0 = np.pi * np.array([1, 1, 1])
        self.b1 = np.pi * np.array([-1, 1, 1])
        self.b2 = np.pi * np.array([1, -1, 1])
        self.b3 = np.pi * np.array([1, 1, -1])

        self.e0 = np.array([0, 0, 0])/2
        self.e1 = np.array([0, 1, 1])/2
        self.e2 = np.array([1, 0, 1])/2
        self.e3 = np.array([1, 1, 0])/2

        self.minLams = np.zeros(2)

        self.Gamma = np.array([0, 0, 0])
        self.L = np.pi * np.array([1, 1, 1])
        self.X = 2*np.pi * np.array([0, 1, 0])
        self.W = 2*np.pi * np.array([0, 1, 1 / 2])
        self.K = 2*np.pi * np.array([0, 3 / 4, 3 / 4])
        self.U = 2*np.pi * np.array([1 / 4, 1, 1 / 4])

        self.res =res
        self.bigB = genBZ(res)

        self.M = np.zeros((2,len(self.bigB)))

        self.GammaX = self.drawLine(self.Gamma, self.X, res)
        self.XW = self.drawLine(self.X, self.W, res)
        self.WK = self.drawLine(self.W, self.K, res)
        self.KGamma = self.drawLine(self.K, self.Gamma, res)
        self.GammaL = self.drawLine(self.Gamma, self.L, res)
        self.LU = self.drawLine(self.L, self.U, res)
        self.UW = self.drawLine(self.U,self.W, res)

        self.dGammaX = np.zeros((2,res))
        self.dXW = np.zeros((2,res))
        self.dWK = np.zeros((2,res))
        self.dKGamma = np.zeros((2,res))
        self.dGammaL = np.zeros((2,res))
        self.dLU = np.zeros((2,res))
        self.dUW = np.zeros((2,res))

        self.V = 1

        self.didit = False

#pyrochlore brillouin zone

#basis vector


    #Critcal Points

    def repcoord(self, a, b, c):
        return a*self.b1+b*self.b2+c*self.b3







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
            temp += 1/2 * self.h * self.neta(alpha) * np.dot(self.n, z(mu)) * np.cos(np.dot(k, self.neta(alpha)*self.NNtest(mu)))
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
        M[:, 1, 0] = np.conj(M[:, 0, 1])
        E, V = np.linalg.eig(M)

        return E

    def minLam(self, alpha):
        # k = obliqueProj(k)
        temp = self.M_zero(self.GammaX[0],alpha)
        if temp == 0:
            temp = -1000
        self.minLams[alpha] = - temp
        return 0

    def setM(self):
        self.M = np.real(self.M_tot(self.bigB).T)
        return 1

    def setupALL(self):
        self.setM()
        self.minLam(0)
        self.minLam(1)
        return 0



    def E_zero_fixed(self,alpha, lams):
        val = np.sqrt(2 * self.Jzz * self.eta[1-alpha] * (lams[alpha] + self.M[alpha]))
        return np.real(val)

    def E_zero(self, k, alpha, lams):
        val = np.sqrt(2 * self.Jzz * self.eta[1-alpha] * (lams[alpha] + self.M_tot(k)[:, alpha]))
        return np.real(val)




    def rho_zero(self, alpha, lams):
        if (self.E_zero_fixed(alpha, lams) == 0).any():
            return 0
        temp = np.mean(self.Jzz*self.eta[alpha]/self.E_zero_fixed(alpha, lams))
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
                 if rhoguess - self.kappa > 0:
                     lamMin = self.lams[alpha]
                 else:
                     lamMax = self.lams[alpha]
             except Exception as e:
                 # print(e)
                 lamMin = self.lams[alpha]
             # print([self.lams[alpha], rhoguess, lamMax, self.minLams[alpha]])
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
        plt.plot(np.linspace(-0.5, 0, self.res), self.dGammaX[alpha], 'b')
        plt.plot(np.linspace(0, 0.3, self.res), self.dXW[alpha] , 'b')
        plt.plot(np.linspace(0.3, 0.5, self.res), self.dWK[alpha], 'b')
        plt.plot(np.linspace(0.5, 1, self.res), self.dKGamma[alpha], 'b')
        plt.plot(np.linspace(1, 1.4, self.res), self.dGammaL[alpha], 'b')
        plt.plot(np.linspace(1.4, 1.7, self.res), self.dLU[alpha], 'b')
        plt.plot(np.linspace(1.7, 1.85, self.res),self.dUW[alpha], 'b')
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

    def graph_even(self, alpha, show):
        if not self.didit:
            self.calDispersion()
        plt.plot(np.linspace(-0.5, 0, self.res), self.dGammaX[alpha], 'b')
        plt.plot(np.linspace(0, 0.5, self.res), self.dXW[alpha] , 'b')
        plt.plot(np.linspace(0.5, 1, self.res), self.dWK[alpha], 'b')
        plt.plot(np.linspace(1, 1.5, self.res), self.dKGamma[alpha], 'b')
        plt.plot(np.linspace(1.5, 2, self.res), self.dGammaL[alpha], 'b')
        plt.plot(np.linspace(2, 2.5, self.res), self.dLU[alpha], 'b')
        plt.plot(np.linspace(2.5, 3, self.res),self.dUW[alpha], 'b')
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



