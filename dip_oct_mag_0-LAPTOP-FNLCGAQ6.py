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
        return np.array([-1, -1, -1])/np.sqrt(3)
    if mu == 1:
        return np.array([-1, 1, 1])/np.sqrt(3)
    if mu == 2:
        return np.array([1, -1, 1])/np.sqrt(3)
    if mu == 3:
        return np.array([1, 1, -1])/np.sqrt(3)

class zeroFluxSolver:

    def __init__(self, Jzz, Jpm, h, n, eta, kappa, lam=20, res=20):
        self.Jzz = Jzz
        self.Jpm = Jpm
        self.kappa = kappa
        self.eta = eta
        self.h = h
        self.n = n

        self.tol = 1e-3
        self.lamA = lam
        self.lamB = lam

        self.b0 = np.pi * np.array([1, 1, 1])
        self.b1 = np.pi * np.array([-1, 1, 1])
        self.b2 = np.pi * np.array([1, -1, 1])
        self.b3 = np.pi * np.array([1, 1, -1])

        self.e0 = np.array([0, 0, 0])/2
        self.e1 = np.array([0, 1, 1])/2
        self.e2 = np.array([1, 0, 1])/2
        self.e3 = np.array([1, 1, 0])/2

        self.Gamma = np.array([0, 0, 0])
        self.minLams = np.zeros(2)

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

    def exponent_mag(self, k, alpha, mu):
        temp = 1/4* self.h* self.neta(alpha) *np.dot(self.n, z(mu))*np.exp(1j*np.dot(k, self.neta(alpha)*self.NNtest(mu)))
        tempc = np.conj(temp)
        temp = temp + tempc
        return temp

    def M_zero(self, k, alpha):
        temp = 0
        for i in range(4):
            for j in range(4):
                if not i==j:
                    temp += self.exponent_zero(k, alpha, i, j)
        if alpha == 0:
            temp = -self.Jpm*self.eta/4 * temp
        else:
            temp = -self.Jpm/4 * temp

        for i in range(4):
            temp += -self.exponent_mag(k, alpha, i)
        return np.real(temp)


    def minLam(self, alpha):
        # k = obliqueProj(k)
        temp = self.M_zero(self.GammaX[0],alpha)
        if temp == 0:
            temp = -1000
        self.minLams[alpha] = - temp
        return 0

    def setM(self):
        for i in range(len(self.bigB)):
            self.M[0][i] = self.M_zero(self.bigB[i], 0)
            self.M[1][i] = self.M_zero(self.bigB[i], 1)
        return 1

    def setupALL(self):
        self.setM()
        self.minLam(0)
        self.minLam(1)
        return 0

    # d = M_pi(np.array([1,0,0]),1)

    #Here we are set for M, but now we also need to solve the
    #self consistency equation for the lagrangian multiplier.


    def E_zero_fixed(self,alpha, LambdaA, LambdaB):
        if alpha == 0:
            try:
                val = np.sqrt(2*self.Jzz *(LambdaA + self.M[alpha]))
                return val
            except:
                return 0
        else:
            try:
                val = np.sqrt(2 * self.Jzz * self.eta * (LambdaB + self.M[alpha]))
                return val
            except:
                return 0

    def E_zero(self, k,alpha, LambdaA, LambdaB):
        if alpha == 0:
            val = np.sqrt(2*self.Jzz *(LambdaA + self.M_zero(k, alpha)))
            return val
        else:
            val = np.sqrt(2 * self.Jzz * self.eta * (LambdaB + self.M_zero(k, alpha)))
            return val



    def rho_zero(self, alpha, LambdaA, LambdaB):
        temp = 0
        eta = 1
        if alpha == 1:
            eta = self.eta
        if eta == 0:
            return 0
        temp = np.mean(self.Jzz*eta/self.E_zero_fixed(alpha, LambdaA, LambdaB))
        return temp



    def findLambda_zero(self,alpha):
        warnings.filterwarnings("error")
        lamMin = 0
        lamMax = 100
        rhoguess = 10
        lam = 0
        while(np.absolute(rhoguess-self.kappa) >= self.tol):
             lam = (lamMin+lamMax)/2
             try:
                 if alpha == 0:
                     rhoguess = self.rho_zero(alpha, lam, self.lamB)
                 else:
                     rhoguess = self.rho_zero(alpha, self.lamA, lam)
                 if lam == 0:
                     return -1000
                 if rhoguess - self.kappa > 0:
                     lamMin = lam
                 else:
                     lamMax = lam
             except:
                 lamMin = lam
             # print([lam, rhoguess])
             if lamMax < self.minLams[alpha]:
                return -1000
        return lam


    #graphing BZ

    def drawLine(self, A, B, N):
        return np.linspace(A, B, N)

    def dispersion_zero(self, P, alpha, lamA, lamB):
        temp = []
        for i in P:
            temp += [self.E_zero(i, alpha, lamA, lamB)]
        return temp

    def findLambda(self):
        self.lamA = self.findLambda_zero(0)
        self.lamB = self.findLambda_zero(1)
        return 1

    def calDispersionA(self, alpha):
        self.dGammaX[alpha] = self.dispersion_zero(self.GammaX, alpha, self.lamA, self.lamB)
        self.dXW[alpha] = self.dispersion_zero(self.XW, alpha, self.lamA, self.lamB)
        self.dWK[alpha] = self.dispersion_zero(self.WK, alpha, self.lamA, self.lamB)
        self.dKGamma[alpha] = self.dispersion_zero(self.KGamma, alpha, self.lamA, self.lamB)
        self.dGammaL[alpha] = self.dispersion_zero(self.GammaL, alpha, self.lamA, self.lamB)
        self.dLU[alpha] = self.dispersion_zero(self.LU, alpha, self.lamA, self.lamB)
        self.dUW[alpha] = self.dispersion_zero(self.UW, alpha, self.lamA, self.lamB)

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
        B = genBZ(20)
        if alpha == 0:
            return np.mean(self.E_zero_fixed(alpha, self.lamA, self.lamB)) - self.lamA
        else:
            return np.mean(self.E_zero_fixed(alpha, self.lamA, self.lamB)) - self.lamB

    def EMAX(self, alpha):
        if not self.didit:
            self.calDispersion()
        return max([max(self.dGammaX[alpha]), max(self.dXW[alpha]), max(self.dWK[alpha]), max(self.dKGamma[alpha]), max(self.dGammaL[alpha]), max(self.dLU[alpha]), max(self.dUW[alpha])])



