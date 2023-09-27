import matplotlib.pyplot as plt
import warnings
from misc_helper import *


piunitcell_here = np.array([
    [[1,0],
     [0,1]],
    [[1,0],
     [0,1]],
    [[0,1],
     [1,0]],
    [[0,1],
     [1,0]],
])

A_pi_here_down = np.array([
    [0,0, np.pi,0],
    [0,np.pi, 0,0]
])

A_pi_here_up = np.array([
    [0,0,0,0],
    [0,np.pi,np.pi,0]
])


A_pi_rs_traced_here_up = np.zeros((2,4,4))
A_pi_rs_traced_here_down = np.zeros((2,4,4))

for i in range(2):
    for j in range(4):
        for k in range(4):
            A_pi_rs_traced_here_up[i,j,k] = np.real(np.exp(1j * (A_pi_here_up[i,j] - A_pi_here_up[i,k])))
            A_pi_rs_traced_here_down[i,j,k] = np.real(np.exp(1j * (A_pi_here_down[i,j] - A_pi_here_down[i,k])))

def M_pi_mag_sub(k, h, n):
    M = np.zeros((len(k),2,2), dtype=np.complex128)
    M[:, 0, 0] = np.exp(1j*np.dot(k,NN[0])) * np.dot(n, z[0]) + np.exp(1j*np.dot(k,NN[1])) * np.dot(n, z[1])
    M[:, 0, 1] = -np.exp(1j*np.dot(k,NN[2])) * np.dot(n, z[2]) + np.exp(1j*np.dot(k,NN[3])) * np.dot(n, z[3])
    M[:, 1, 0] = np.exp(1j*np.dot(k,NN[2]))*np.dot(n, z[2]) + np.exp(1j*np.dot(k,NN[3])) * np.dot(n, z[3])
    M[:, 1, 1] = np.exp(1j*np.dot(k,NN[0]))*np.dot(n, z[0]) - np.exp(1j*np.dot(k,NN[1]))*np.dot(n, z[1])
    return -h/4*M


# def M_pi_sub_0(k, Jpm):
#     kx = k[:,0]
#     ky = k[:,1]
#     kz = k[:,2]
#     M = np.zeros((len(k),2,2), dtype=np.complex128)
#     M[:,0,0] = np.multiply(np.cos(ky/2), np.cos(kz/2))
#     M[:, 0, 1] = -np.multiply(np.sin(kx / 2), np.sin(ky / 2)) - 1j*np.multiply(np.cos(kx/2),np.sin(kz/2))
#     M[:, 1, 0] = -np.multiply(np.sin(kx / 2), np.sin(ky / 2)) + 1j*np.multiply(np.cos(kx/2),np.sin(kz/2))
#     M[:, 1, 1] = -np.multiply(np.cos(ky/2), np.cos(kz/2))
#     return Jpm*M
#
# def M_pi_sub_1(k, Jpm):
#     k = k+np.pi*np.array([1,1,1])
#     kx = k[:,0]
#     ky = k[:,1]
#     kz = k[:,2]
#     M = np.zeros((len(k),2,2), dtype=np.complex128)
#     M[:,0,0] = np.multiply(np.cos(ky/2), np.cos(kz/2))
#     M[:, 0, 1] = np.multiply(np.sin(kx / 2), np.sin(ky / 2)) - 1j*np.multiply(np.cos(kx/2),np.sin(kz/2))
#     M[:, 1, 0] = np.multiply(np.sin(kx / 2), np.sin(ky / 2)) + 1j*np.multiply(np.cos(kx/2),np.sin(kz/2))
#     M[:, 1, 1] = -np.multiply(np.cos(ky/2),np.cos(kz/2))
#     return Jpm*M

def M_pi_mag_sub_AB(k, h, n, theta):
    zmag = contract('k,ik->i',n,z)
    ffact = contract('ik, jk->ij', k, NN)
    ffact = np.exp(1j*ffact)
    M = contract('ku, u, ru, urx->krx',-1/4*h*ffact*(np.cos(theta)-1j*np.sin(theta)), zmag, np.exp(1j*A_pi), piunitcell)
    return M


def M_pi_sub_intrahopping_BB(k, alpha, eta, Jpm):
    ffact = contract('ik, jlk->ijl', k,NNminus)
    ffact = np.exp(-1j*neta(alpha)*ffact)
    M = contract('jl,kjl,ijl, jka, lkb->iab', notrace, -Jpm*A_pi_rs_traced/4*eta[alpha], ffact, piunitcell, piunitcell)
    return M


def M_pi(k,eta,Jpm, h, n):
    # M1 = M_pi_sub_0(k,Jpm)
    # M2 = M_pi_sub_1(k,Jpm)
    M1 =M_pi_sub_intrahopping_BB(k, 1, eta, Jpm)
    M2 = M_pi_sub_intrahopping_BB(k, 0, eta, Jpm)
    Mag1 = M_pi_mag_sub_AB(k, h, n, 0)
    # temp = M_pi_mag_sub(k,h,n)
    Mag2 = np.conj(np.transpose(Mag1, (0,2,1)))
    FM = np.block([[M1, Mag1], [Mag2, M2]])
    return FM


def E_pi_fixed(lams, M):
    M = M + np.diag(np.repeat(lams,4))
    E, V = np.linalg.eigh(M)
    return [E, V]



def E_pi(lams, k, eta, Jpm, h, n):
    M = M_pi(k,eta,Jpm, h, n)
    M = M + np.diag(np.repeat(lams,4))
    E, V = np.linalg.eigh(M)
    # print(E)
    return [E,V]


def rho_true(Jzz, M, lams):
    # dumb = np.array([[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1]])
    temp = M + np.diag(np.repeat(lams,4))
    E, V = np.linalg.eigh(temp)
    Vt = np.real(contract('ijk,ijk->ijk',V, np.conj(V)))
    Ep = contract('ijk, ik->ij', Vt, Jzz/np.sqrt(2*Jzz*E))
    return np.mean(Ep)*np.ones(2)


def findminLam(M, Jzz, tol):
    warnings.filterwarnings("error")
    lamMin = np.zeros(2)
    lamMax = 50*np.ones(2)
    lams = (lamMin + lamMax) / 2

    while ((lamMax-lamMin>=tol).all()):
        lams = (lamMin + lamMax) / 2
        try:
             rhoguess = rho_true(Jzz, M, lams)
             for i in range(2):
                 lamMax[i] = lams[i]
        except:
             lamMin = lams
        # print([lams, lamMin, lamMax,lamMax-lamMin])

    return lams

def findlambda_pi(M, Jzz, kappa, tol):
    warnings.filterwarnings("error")
    lamMin = np.zeros(2)
    lamMax = 50*np.ones(2)
    lams = (lamMin + lamMax) / 2
    rhoguess = rho_true(Jzz, M, lams)
    # print(self.kappa)

    while not ((np.absolute(rhoguess-kappa)<=tol).all()):
        # for i in range(2):
         lams= (lamMin+lamMax)/2
         try:
             rhoguess = rho_true(Jzz, M, lams)
             # rhoguess = self.rho_zero(alpha, self.lams)
             for i in range(2):
                 if rhoguess[i] - kappa > 0:
                     lamMin[i]  = lams[i]
                 else:
                     lamMax[i]  = lams[i]
         except:
             lamMin = lams
             # if lamMax == 0:
             #     break
         # print([lams, rhoguess])
         # if (lamMax - lamMin <= tol ** 2 * 10).all():
         #     lams = -1000 * np.ones(2)
    return lams



# graphing BZ

def dispersion_pi(lams, k, Jzz, Jpm, eta, h, n):
    temp = np.sqrt(2*Jzz*E_pi(lams, k, eta, Jpm, h, n)[0])
    return temp

def calDispersion(lams, Jzz, Jpm, eta, h, n):
    dGammaX= dispersion_pi(lams, GammaX, Jzz, Jpm, eta, h, n)
    dXW= dispersion_pi(lams, XW, Jzz, Jpm, eta, h, n)
    dWK = dispersion_pi(lams, WK, Jzz, Jpm, eta, h, n)
    dKGamma = dispersion_pi(lams, KGamma, Jzz, Jpm, eta, h, n)
    dGammaL = dispersion_pi(lams, GammaL, Jzz, Jpm, eta, h, n)
    dLU= dispersion_pi(lams, LU, Jzz, Jpm, eta, h, n)
    dUW = dispersion_pi(lams, UW, Jzz, Jpm, eta, h, n)

    for i in range(8):
        plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX[:,i], 'b')
        plt.plot(np.linspace(gX, gW1, len(dXW)), dXW[:, i] , 'b')
        plt.plot(np.linspace(gW1, gK, len(dWK)), dWK[:, i], 'b')
        plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma[:, i], 'b')
        plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL[:, i], 'b')
        plt.plot(np.linspace(gL, gU, len(dLU)), dLU[:, i], 'b')
        plt.plot(np.linspace(gU, gW2, len(dUW)),dUW[:, i], 'b')

    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=gGamma1, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW1, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gK, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma2, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gL, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gU, color='b', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW2, color='b', label='axvline - full height', linestyle='dashed')
    xlabpos = [gGamma1,gX,gW1,gK,gGamma2,gL,gU,gW2]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)

# @nb.njit(parallel=True, cache=True)
def minCal(lams, q, Jzz, Jpm, eta, h, n, K):
    temp = np.zeros(len(q))
    mins = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, h, n)[0])[:,0]
    for i in range(len(q)):
        temp[i] = np.min(np.sqrt(2 * Jzz * E_pi(lams, K-q[i], eta, Jpm, h, n)[0])[:,0] + mins)
    return temp

# @nb.njit(parallel=True, cache=True)
def maxCal(lams, q, Jzz, Jpm, eta, h, n, K):
    temp = np.zeros(len(q))
    maxs = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, h, n)[0])[:,-1]
    for i in range(len(q)):
        temp[i] = np.max(np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, h, n)[0])[:,-1] + maxs)
    return temp

def minMaxCal(lams, q, Jzz, Jpm, eta, h, n, K):
    temp = np.zeros((len(q),2))
    maxs = np.sqrt(2 * Jzz * E_pi(lams, K, eta, Jpm, h, n)[0])
    for i in range(len(q)):
        stuff = np.sqrt(2 * Jzz * E_pi(lams, K-q[i], eta, Jpm, h, n)[0]) + maxs
        temp[i,0] = np.min(stuff[:,0])
        temp[i,1] = np.max(stuff[:,-1])
    return temp

def loweredge(lams, Jzz, Jpm, eta, h, n, K):
    dGammaX= minCal(lams, GammaX, Jzz, Jpm, eta, h, n, K)
    dXW= minCal(lams, XW, Jzz, Jpm, eta, h, n, K)
    dWK = minCal(lams, WK, Jzz, Jpm, eta, h, n, K)
    dKGamma = minCal(lams, KGamma, Jzz, Jpm, eta, h, n, K)
    dGammaL = minCal(lams, GammaL, Jzz, Jpm, eta, h, n, K)
    dLU= minCal(lams, LU, Jzz, Jpm, eta, h, n, K)
    dUW = minCal(lams, UW, Jzz, Jpm, eta, h, n, K)

    plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, 'b')
    plt.plot(np.linspace(gX, gW1, len(dXW)), dXW, 'b')
    plt.plot(np.linspace(gW1, gK, len(dWK)), dWK, 'b')
    plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, 'b')
    plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, 'b')
    plt.plot(np.linspace(gL, gU, len(dLU)), dLU, 'b')
    plt.plot(np.linspace(gU, gW2, len(dUW)),dUW, 'b')

    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW2, color='w', label='axvline - full height', linestyle='dashed')
    xlabpos = [gGamma1,gX,gW1,gK,gGamma2,gL,gU,gW2]
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

    plt.plot(np.linspace(gGamma1, gX, len(dGammaX)), dGammaX, 'b')
    plt.plot(np.linspace(gX, gW1, len(dXW)), dXW, 'b')
    plt.plot(np.linspace(gW1, gK, len(dWK)), dWK, 'b')
    plt.plot(np.linspace(gK, gGamma2, len(dKGamma)), dKGamma, 'b')
    plt.plot(np.linspace(gGamma2, gL, len(dGammaL)), dGammaL, 'b')
    plt.plot(np.linspace(gL, gU, len(dLU)), dLU, 'b')
    plt.plot(np.linspace(gU, gW2, len(dUW)),dUW, 'b')

    plt.ylabel(r'$\omega/J_{zz}$')
    plt.axvline(x=gGamma1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gX, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW1, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gK, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gGamma2, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gL, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gU, color='w', label='axvline - full height', linestyle='dashed')
    plt.axvline(x=gW2, color='w', label='axvline - full height', linestyle='dashed')
    xlabpos = [gGamma1,gX,gW1,gK,gGamma2,gL,gU,gW2]
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', r'$L$', r'$U$', r'$W$']
    plt.xticks(xlabpos, labels)

def gap(M, lams):
    temp = M + np.diag(np.repeat(lams,4))
    E,V = np.linalg.eigh(temp)
    # E = np.sqrt(E)
    temp = np.amin(E)
    return temp

def EMAX(M, lams):
    temp = M + np.diag(np.repeat(lams,4))
    E,V = np.linalg.eigh(temp)
    temp = np.amax(E)
    return temp

def green_pi(E, V, Jzz):
    Vt = contract('ijk, ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    # temp = 2*pypi.Jzz*np.multiply(pypi.V[:,nu,i], np.conj(np.transpose(pypi.V, (0, 2, 1)))[:,i,mu])
    green = Jzz/np.sqrt(2*Jzz*E)
    green = contract('ikjl, ik->ijl', Vt, green)
    return green

def green_pi_old(E, V, Jzz):
    Vt = contract('ijk, ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    green = Jzz/np.sqrt(2*Jzz*E)
    green = contract('ikjl, ik->ijl', Vt, green)
    return green

def green_pi_branch(E, V, Jzz):
    Vt = contract('ijk, ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    green = Jzz/np.sqrt(2*Jzz*E)
    green = contract('ikjl, ik->ikjl', Vt, green)
    return green

def green_pi_old_branch(k, E, V, Jzz):
    Vt = contract('ijk, ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    green = Jzz/np.sqrt(2*Jzz*E)
    green = contract('ikjl, ik->ikjl', Vt, green)
    return green


#
# def GS(lams, k, Jzz, Jpm, eta, h, n):
#     return np.mean(dispersion_pi(lams, k, Jzz, Jpm, eta), axis=0) - np.repeat(lams)

class piFluxSolver:

    def __init__(self, Jpm, h=0, n=np.array([0,0,0]), eta=1, kappa=1, lam=2, BZres=20, graphres=20, Jzz=1):
        self.Jzz = Jzz
        self.Jpm = Jpm
        self.kappa = kappa
        self.eta = np.array([eta, 1], dtype=float)
        self.tol = 1e-5
        self.lams = np.array([lam, lam], dtype=float)
        self.h = h
        self.n = n


        self.minLams = np.zeros(2)

        self.BZres = BZres
        self.graphres = graphres
        self.bigB = np.concatenate((genBZ(BZres), symK))
        self.MF = M_pi(self.bigB, self.eta, self.Jpm, self.h, self.n)

        self.q = np.empty((len(self.bigB),3))
        self.q[:] = np.nan

    #alpha = 1 for A = -1 for B


    def findLambda(self):
        self.lams = findlambda_pi(self.MF, self.Jzz, self.kappa, self.tol)
        warnings.resetwarnings()

    def findminLam(self):
        self.minLams = findminLam(self.MF, self.Jzz, 1e-10)
        warnings.resetwarnings()

    def qvec(self):
        # print((2e2/len(self.bigB))**2)
        E = E_pi(self.lams-np.ones(2)*(2e2/len(self.bigB))**2, self.bigB, self.eta, self.Jpm, self.h, self.n)[0]
        c = np.unique(np.where(E < 0)[0])
        temp = np.unique(self.bigB[c], axis=0)
        self.q[0:len(temp),:] = temp

    def ifcondense(self, q):
        E = E_pi(self.lams-np.ones(2)*(2e2/len(self.bigB))**2, q, self.eta, self.Jpm, self.h, self.n)[0]
        c = np.unique(np.where(E < 0)[0])
        return c

    def condensed(self):
        return np.absolute(self.minLams - self.lams) < (2e2/len(self.bigB))**2

    def M_true(self, k):
        return M_pi(k, self.eta, self.Jpm, self.h, self.n)

    # def M_pi_sub(self, k, rs, alpha):
    #     return M_pi_sub(k, rs, alpha, self.eta, self.Jpm)

    def E_pi(self, k):
        return np.sqrt(2*self.Jzz*E_pi(self.lams, k, self.eta, self.Jpm, self.h, self.n)[0])

    def dispersion(self, k):
        return dispersion_pi(self.lams, k, self.Jzz, self.Jpm, self.eta, self.h, self.n)

    def LV_zero(self, k, lam=np.zeros(2)):
        if np.any(lam == 0):
            lam = self.lams
        return E_pi(lam, k, self.eta, self.Jpm, self.h, self.n)

    # def LV_zero_old(self, k,alpha):
    #     M = M_pi(k, alpha, self.eta, self.Jpm)
    #     M = M+np.diag(np.repeat(self.lams[0],4))
    #     E,V = np.linalg.eigh(M)
    #     if alpha == 1:
    #         V = np.conj(V)
    #     return [E,V]

    def GS(self):
        return np.mean(self.E_pi(self.bigB)) - self.lams[0]


    # def LV_zero_old_single(self, k, alpha):
    #     M = M_pi_single(k, self.eta, self.Jpm, self.h, self.n)[4*alpha:4*(alpha+1), 4*alpha:4*(alpha+1)] + np.diag(np.repeat(self.lams[alpha], 4))
    #     E, V = np.linalg.eigh(M)
    #     return [E,V]

    def gap(self):
        return np.sqrt(2*self.Jzz*gap(self.MF, self.lams))

    def gapwhere(self):
        temp = self.MF + np.diag(np.repeat(self.lams,4))
        E, V = np.linalg.eigh(temp)
        # E = np.sqrt(2*self.Jzz*E)
        dex = np.argmin(E,axis=0)[0]
        return np.mod(self.bigB[dex], 2*np.pi)

    def graph(self, show):
        calDispersion(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n)
        if show:
            plt.show()

    # def M_single(self, k):
    #     M = M_pi_single(k, self.eta, self.Jpm, self.h, self.n) + np.diag(np.repeat(self.lams, 4))
    #     return M
    #
    #
    # def E_single(self, k):
    #     M = M_pi_single(k, self.eta, self.Jpm, self.h, self.n) + np.diag(np.repeat(self.lams, 4))
    #     E, V = np.linalg.eigh(M)
    #     return np.sqrt(2*self.Jzz*E)

    def minCal(self, K):
        return minCal(self.lams, K, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB)

    def maxCal(self, K):
        return maxCal(self.lams, K, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB)
    def minMaxCal(self, K):
        return minMaxCal(self.lams, K, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB)

    def EMAX(self):
        return np.sqrt(2*self.Jzz*EMAX(self.MF, self.lams))

    def TWOSPINON_GAP(self, k):
        return np.min(minCal(self.lams, k, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB))

    def TWOSPINON_MAX(self, k):
        return np.max(maxCal(self.lams, k, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB))


    def graph_loweredge(self, show):
        loweredge(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB)
        if show:
            plt.show()

    def graph_upperedge(self, show):
        upperedge(self.lams, self.Jzz, self.Jpm, self.eta, self.h, self.n, self.bigB)
        if show:
            plt.show()

    # def green_pi(self, k):
    #     E, V = self.LV_zero(k)
    #     return green_pi(E, V, self.Jzz)
    #
    # def green_pi_branch(self, k):
    #     E, V = self.LV_zero(k)
    #     return green_pi_branch(E, V, self.Jzz), E
    #
    # def green_pi_old(self, k, alpha):
    #     E, V = self.LV_zero_old(k,alpha)
    #     return green_pi_old(k, E, V, self.Jzz)
    #
    # def green_pi_old_branch(self, k, alpha):
    #     E, V = self.LV_zero_old(k,alpha)
    #     return green_pi_old_branch(k, E, V, self.Jzz)

    def green_pi(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k, lam)
        E = np.sqrt(2 * self.Jzz * E)
        return green_pi(E, V, self.Jzz)

    def green_pi_branch(self, k, lam=np.zeros(2)):
        E, V = self.LV_zero(k,lam)
        E = np.sqrt(2*self.Jzz*E)
        return green_pi_branch(E, V, self.Jzz), E

    def magnetization(self):
        green = self.green_pi(self.bigB)
        zmag = contract('k,ik->i', self.n, z)
        ffact = contract('ik, jk->ij', k, NN)
        ffactp = np.exp(1j * ffact)
        ffactm = np.exp(-1j * ffact)

        magp = contract('j, ij, ika, ikj, jka->i', zmag, ffactp, green[:, 0:4, 4:8], np.exp(1j * A_pi),
                        piunitcell) / 4
        magm = contract('j, ij, iak, ikj, jka->i', zmag, ffactm, green[:, 4:8, 0:4], np.exp(-1j * A_pi),
                        piunitcell) / 4

        return np.real(np.mean(magp + magm)) / 4