
import numpy as np
from pyrochlore_dispersion_pi import *
from numpy import ndarray
# from mpi4py import MPI
import time
import math
import sys
from numba import jit

class myarray(ndarray):
    @property
    def H(self):
        return self.conj().T

def NN(mu):
    if mu == 0:
        return np.array([-1 / 4, -1 / 4, -1 / 4])
    if mu == 1:
        return np.array([-1 / 4, 1 / 4, 1 / 4])
    if mu == 2:
        return np.array([1 / 4, -1 / 4, 1 / 4])
    if mu == 3:
        return np.array([1 / 4, 1 / 4, -1 / 4])



def green_zero(k, omega, alpha, pyp0):
    return 2*pyp0.Jzz/(omega**2 + 2*pyp0.Jzz*(pyp0.lams[alpha]+pyp0.E_zero(k,alpha, pyp0.lams)))



def green_pi(k, omega, alpha, pypi, mu, nu, i):
    temp = 2*pypi.Jzz*np.multiply(pypi.V[:,nu,i], np.conj(np.transpose(pypi.V, (0, 2, 1)))[:,i,mu])
    tempc = np.divide(temp, omega ** 2 + 2 * pypi.Jzz * (pypi.lams[alpha] + pypi.E_pi(k, alpha, pypi.lams)[:, i]))
    return tempc

def gaussian(mu, tol):
    return np.exp(-np.power( - mu, 2) / (2 * np.power(tol, 2)))

def cauchy(mu, tol):
    return tol/(mu**2+tol**2)/np.pi

def formfactor():
    M=np.zeros((4,4,3))
    for i in range(4):
        for j in range(4):
            M[i][j] = NN(i)-NN(j)
    return M

def spinon_cont_zero(q, omega, alpha, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks+q
    le = len(Ks)


    tempE= pyp0.E_zero(Ks, alpha, pyp0.lams)
    tempQ = pyp0.E_zero(Qs, alpha, pyp0.lams)
    M = formfactor()

    ffac = np.einsum('ij, klj -> ikl', Qs, M)
    ffac = np.exp(1j*ffac)
    ffac = np.einsum('ikl->i', ffac)


    green = np.multiply(green_zero(Ks, omega, alpha, pyp0), green_zero(Qs, omega, alpha, pyp0))
    temp = np.multiply(cauchy(omega-tempE-tempQ, tol), green)
    inte = np.multiply(temp, ffac)



    return np.real(np.sum(inte))

def SSSF_zero(q, alpha, pyp0):
    Ks = pyp0.bigB
    Qs = Ks+q
    le = len(Ks)

    M = formfactor()

    ffac = np.einsum('ij, klj -> ikl', Qs, M)
    ffac = np.exp(1j*ffac)
    ffac = np.einsum('ikl->i', ffac)


    green = np.multiply(green_zero(Ks, 0, alpha, pyp0), green_zero(Qs, 0, alpha, pyp0))
    inte = np.multiply(green, ffac)

    return np.real(np.sum(inte))


def spinon_cont_pi(q, omega, alpha, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks+q

    tempE = pyp0.E_pi(Ks, alpha, pyp0.lams)
    tempQ = pyp0.E_pi(Qs, alpha, pyp0.lams)
    s = np.array([1,1,1,1])
    dumb = np.einsum('i,j->ij', s, s)
    tempE = np.einsum('ij,k->ijk', tempE, s)
    tempQ = np.transpose(np.einsum('ij,k->ijk', tempQ, s), (0,2,1))

    gauss = gaussian(omega - tempE - tempQ, tol)

    M = formfactor()
    ffac = np.einsum('ij, klj -> ikl', Qs, M)
    ffac = np.exp(1j*ffac)

    greenp1 = np.zeros((len(Ks),4,4), dtype=complex)
    greenp2 = np.zeros((len(Ks),4,4,4,4), dtype=complex)

    # vgreenpi = np.vectorize(green_pi, excluded=['omega', 'alpha', 'pypi', 'mu', 'nu', 'i'])
    for rs in range(4):
        for gamma in range(4):
            for i in range(4):
                for j in range(4):
                    if not i == j:
                        mu = unitCell(rs) + neta(alpha) * step(i)
                        nu = unitCell(rs) + neta(alpha) * step(j)
                        rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                        rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                        index1 = findS(rs1)
                        index2 = findS(rs2)
                        greenp1[:, gamma, rs] = green_pi(Ks,omega,alpha, pyp0, rs, rs, gamma)
                        greenp2[:, gamma, rs, i, j] = green_pi(Ks, omega, alpha, pyp0, index1, index2, gamma)

    greenp1 = np.einsum('ijk, la-> ijkla', greenp1, dumb)
    temp = np.einsum('ijakl, ikl-> ijakl', greenp2, ffac)
    temp2 = np.einsum('iajkl, ibjlo-> iab', greenp1, temp)

    inte = np.einsum('ijk, ijk', temp2, gauss)
    return np.real(inte)


def SSSF_pi(q, alpha, pyp0):
    Ks = pyp0.bigB
    Qs = Ks+q

    s = np.array([1,1,1,1])
    dumb = np.einsum('i,j->ij', s, s)

    M = formfactor()
    ffac = np.einsum('ij, klj -> ikl', Qs, M)
    ffac = np.exp(1j*ffac)

    greenp1 = np.zeros((len(Ks),4), dtype=complex)
    greenp2 = np.zeros((len(Ks),4,4,4), dtype=complex)

    for rs in range(4):
        for gamma in range(4):
            for i in range(4):
                for j in range(4):
                    if not i == j:
                        mu = unitCell(rs) + neta(alpha) * step(i)
                        nu = unitCell(rs) + neta(alpha) * step(j)
                        rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                        rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                        index1 = findS(rs1)
                        index2 = findS(rs2)
                        greenp1[:, rs] = green_pi(Ks,0,alpha, pyp0, rs, rs, gamma)
                        greenp2[:, rs, i, j] = green_pi(Ks, 0, alpha, pyp0, index1, index2, gamma)
    greenp1 = np.einsum('ijk, la-> ijkla', greenp1, dumb)
    temp = np.einsum('ijakl, ikl-> ijakl', greenp2, ffac)
    inte = np.einsum('iajkl, ibjlo', greenp1, temp)

    return np.real(inte)


def graph_spin_cont_pi(pyp0, E, K, tol):
    el = "==:==:=="
    totaltask = len(E)*len(K)
    increment = totaltask/50
    count = 0

    temp = np.zeros((len(E), len(K)))

    for i in range(len(E)):
        for j in range(len(K)):
            start = time.time()
            count = count + 1
            temp[i][j] = spinon_cont_pi(K[j], E[i], 0, pyp0, tol)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            end = time.time()
            el = (end - start)*(totaltask-count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            sys.stdout.flush()
    return temp/np.max(temp)
    # E, K = np.meshgrid(e, K)


def graph_spin_cont_zero(pyp0, E, K, tol):
    el = "==:==:=="
    totaltask = len(E)*len(K)
    increment = totaltask/50
    count = 0
    temp = np.zeros((len(E), len(K)))
    for i in range(len(E)):
        for j in range(len(K)):
            start = time.time()
            count = count + 1
            temp[i][j] = spinon_cont_zero(K[j], E[i], 0, pyp0, tol)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            end = time.time()
            el = (end - start)*(totaltask-count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            sys.stdout.flush()
    return temp/np.max(temp)
    # E, K = np.meshgrid(e, K)


def graph_SSSF_zero(pyp0, K):
    el = "==:==:=="
    totaltask = len(K)
    increment = totaltask/50
    count = 0

    temp = np.zeros(len(K))

    for j in range(len(K)):
        start = time.time()
        count = count + 1
        temp[j] = SSSF_zero(K[j], 0, pyp0)
        # if temp[i][j] > tempMax:
        #     tempMax = temp[i][j]
        end = time.time()
        el = (end - start)*(totaltask-count)
        el = telltime(el)
        sys.stdout.write('\r')
        sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
        sys.stdout.flush()

    return temp/np.max(temp)
    # E, K = np.meshgrid(e, K)


def graph_SSSF_pi(pyp0, K):
    el = "==:==:=="
    totaltask = len(K)
    increment = totaltask/50
    count = 0
    temp = np.zeros(len(K))

    for j in range(len(K)):
        start = time.time()
        count = count + 1
        temp[j] = SSSF_pi(K[j], 0, pyp0)
        # if temp[i][j] > tempMax:
        #     tempMax = temp[i][j]
        end = time.time()
        el = (end - start)*(totaltask-count)
        el = telltime(el)
        sys.stdout.write('\r')
        sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
        sys.stdout.flush()
    return temp/np.max(temp)
    # E, K = np.meshgrid(e, K)





def telltime(sec):
    hours = math.floor(sec/3600)
    sec = sec-hours*3600
    minus = math.floor(sec/60)
    sec = int(sec - minus * 60)
    return str(hours) + ':' + str(minus) + ':' + str(sec)








