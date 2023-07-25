
import numpy as np
from pyrochlore_dispersion_pi import *
from numpy import ndarray
from mpi4py import MPI
import time
import math
import sys

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





def spinon_cont_zero(q, omega, alpha, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks+q
    le = len(Ks)


    tempE= pyp0.E_zero(Ks, alpha, pyp0.lams)
    tempQ = pyp0.E_zero(Qs, alpha, pyp0.lams)

    green = np.multiply(green_zero(Ks, omega, alpha, pyp0), green_zero(Qs, omega, alpha, pyp0))
    inte = np.multiply(gaussian(omega-tempE-tempQ, tol), green)


    return np.real(np.sum(inte))

def spinon_cont_pi(q, omega, alpha, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks+q

    tempE = pyp0.E_pi(Ks, alpha, pyp0.lams)
    tempQ= pyp0.E_pi(Qs, alpha, pyp0.lams)


    inte = np.zeros(len(Ks), dtype=complex)

    # vgreenpi = np.vectorize(green_pi, excluded=['omega', 'alpha', 'pypi', 'mu', 'nu', 'i'])
    for rs in range(4):
        for gamma in range(4):
            for gammap in range(4):
                for i in range(4):
                    for j in range(4):
                        if not i == j:
                            mu = unitCell(rs) + neta(alpha) * step(i)
                            nu = unitCell(rs) + neta(alpha) * step(j)
                            rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                            rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                            index1 = findS(rs1)
                            index2 = findS(rs2)
                            temp = np.multiply(green_pi(Ks, omega, alpha, pyp0, rs, rs, gamma), green_pi(Qs, omega, alpha, pyp0, index1, index2, gammap))
                            inte += np.multiply(gaussian(omega - tempE[:,gamma] - tempQ[:,gammap], tol), temp)

                            # inte += np.multiply(np.multiply(gaussian(omega - tempE[:,gamma] - tempQ[:,gammap], tol), temp),np.exp(-1j*np.dot(Qs, NNtest(i)-NNtest(j))))

    return np.real(np.sum(inte))


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

def telltime(sec):
    hours = math.floor(sec/3600)
    sec = sec-hours*3600
    minus = math.floor(sec/60)
    sec = int(sec - minus * 60)
    return str(hours) + ':' + str(minus) + ':' + str(sec)








