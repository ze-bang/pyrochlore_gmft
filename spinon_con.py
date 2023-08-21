
import numpy as np
from pyrochlore_dispersion_pi import *
from numpy import ndarray
# from mpi4py import MPI
import time
import math
import sys
from numba import jit
from misc_helper import *

class myarray(ndarray):
    @property
    def H(self):
        return self.conj().T




def green_zero(k, pyp0):
    E, V = pyp0.LV_zero(k)
    Vt = np.einsum('ijk,ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    green = pyp0.Jzz/np.sqrt(2*pyp0.Jzz*E)
    green = np.einsum('ikjl, ik->ijl', Vt, green)
    return green

def green_zero_old(k, alpha, omega, pyp0):
    E = pyp0.E_zero_old(k, alpha)
    green = pyp0.Jzz/(omega**2 + E)
    return green

def green_f(k, omega, pyp0):
    M = pyp0.M_true(k)
    M = M + np.diag(pyp0.lams) + np.identity(2)*omega**2/(2*pyp0.Jzz)
    return np.linalg.inv(M)


def green_pi(k, pypi):
    E, V = pypi.LV_zero(k)
    Vt = np.einsum('ijk, ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    # temp = 2*pypi.Jzz*np.multiply(pypi.V[:,nu,i], np.conj(np.transpose(pypi.V, (0, 2, 1)))[:,i,mu])
    green = pypi.Jzz/np.sqrt(2*pypi.Jzz*E)
    green = np.einsum('ikjl, ik->ijl', Vt, green)
    return green

def green_pi_branch(k, pypi):
    E, V = pypi.LV_zero(k)
    Vt = np.einsum('ijk, ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    # temp = 2*pypi.Jzz*np.multiply(pypi.V[:,nu,i], np.conj(np.transpose(pypi.V, (0, 2, 1)))[:,i,mu])
    green = pypi.Jzz/np.sqrt(2*pypi.Jzz*E)
    green = np.einsum('ikjl, ik->ikjl', Vt, green)
    return green



# def green_pi_test_1(k, omega, alpha, pypi):
#     V = pypi.V[alpha]
#     E = pypi.E_pi(k, alpha, pypi.lams)
#     temp = np.zeros((len(k),4,4,4), dtype=complex)
#     for i in range(4):
#         for mu in range(4):
#             for nu in range(4):
#                 if not mu == nu:
#                     temp[:, i, mu, nu] =  E[:, i]
#     return temp

def gaussian(mu, tol):
    return np.exp(-np.power( - mu, 2) / (2 * np.power(tol, 2)))

def cauchy(mu, tol):
    return tol/(mu**2+tol**2)/np.pi

def formfactorpm():
    M=np.zeros((4,4,3))
    for i in range(4):
        for j in range(4):
            M[i][j] = NN(i)-NN(j)
    return M

def formfactorpp(K, Q, alpha):
    M=np.zeros((len(K), 4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            M[:,i,j] = np.exp(-1j*neta(alpha)*np.dot(K, NN(i))) * np.exp(-1j*neta(alpha)*np.dot(Q, NN(j)))
    return M

def formfactormm(K, Q, alpha):
    M=np.zeros((len(K), 4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            M[:,i,j] = np.exp(1j*neta(alpha)*np.dot(Q, NN(i))) * np.exp(1j*neta(alpha)*np.dot(K, NN(j)))
    return M

def gaugefieldpi(alpha, K):
    gauge = np.zeros((len(K), 4,4,4), dtype=complex)
    for rs in range(4):
        for i in range(4):
            for j in range(4):
                mu = unitCell(rs) + neta(alpha) * step(i)
                nu = unitCell(rs) + neta(alpha) * step(j)
                rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                index1 = findS(rs1)
                index2 = findS(rs2)
                gauge[:, rs, index1, index2] += np.exp(1j * neta(alpha) * (A_pi(unitCell(rs), rs2) - A_pi(unitCell(rs), rs1))) \
                * np.exp(-1j * neta(alpha) * np.dot(K, NN(i)-NN(j)))
    return gauge/4

def gaugepi(K):
    gauge = np.zeros((len(K), 8, 8, 8), dtype=complex)
    gauge[:,0:4,0:4,0:4] = gaugefieldpi(0, K)
    gauge[:,4:8,4:8,4:8] = gaugefieldpi(1, K)
    return gauge


def spinon_cont_zero(q, omega, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks-q
    dum = np.array([[1,1],[1,1]])
    le = len(Ks)
    # sQ = np.einsum('i,j->ij', np.ones(le), q)
    #
    tempE= np.einsum('ij, jk->ijk',pyp0.E_zero(Ks),dum)
    tempQ = np.einsum('ij, jk->ikj',pyp0.E_zero(Qs),dum)

    M = formfactorpm()

    ffac = np.einsum('ij, klj -> ikl', Ks, M)
    ffacA = np.exp(-1j*neta(0)*ffac)
    ffacA = np.einsum('ikl->i', ffacA)

    ffacB = np.exp(-1j*neta(1)*ffac)
    ffacB = np.einsum('ikl->i', ffacB)

    green = np.einsum('ijj,ikk->ijk',green_zero(Ks, pyp0), green_zero(Qs, pyp0))
    temp = np.multiply(cauchy(omega-tempE-tempQ, tol), green)
    temp = temp[:,0,1]*(ffacA+np.conj(ffacA))+ temp[:,1,0]*(ffacB+np.conj(ffacB))

    return np.real(np.mean(temp))/2

def SSSF_zero(q, pyp0):
    Ks = pyp0.bigB
    Qs = Ks-q
    le = len(Ks)
    # sQ = np.einsum('i,j->ij', np.ones(le), q)

    greenp1 = green_zero(Ks, pyp0)
    greenp2 = green_zero(Qs, pyp0)

    #region S+-
    M = formfactorpm()
    ffac = np.einsum('ij, klj -> ikl', Ks, M)
    ffacA = np.exp(-1j*neta(0)*ffac)
    ffacA = np.einsum('ikl->i', ffacA)

    ffacB = np.exp(-1j*neta(1)*ffac)
    ffacB = np.einsum('ikl->i', ffacB)

    greenA = np.einsum('ijj,ikk->ijk',greenp1, greenp2)
    greenpm = greenA[:,0,1]*(ffacA+np.conj(ffacA)) + greenA[:,1,0]*(ffacB+np.conj(ffacB))
    #endregion



    #region S++
    SppA = np.einsum('ijk->i',formfactorpp(Ks, Qs, 0))
    SppB = np.einsum('ijk->i',formfactorpp(Ks, Qs, 1))

    greenB = np.einsum('ijk,ikj->ijk', greenp1, greenp2)
    greenpp = greenB[:, 0, 1] * (SppA + np.conj(SppA)) + greenB[:, 1, 0] * (SppB + np.conj(SppB))

    return np.real(np.mean(greenpm+greenpp)/4)/2


def spinon_cont_pi(q, omega, alpha, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks-q

    tempE = pyp0.E_pi(Ks, alpha, pyp0.lams)
    tempQ = pyp0.E_pi(Qs, alpha, pyp0.lams)

    s = np.array([1,1,1,1,1,1,1,1])
    tempE = np.einsum('ij,k->ijk', tempE, s)
    tempQ = np.einsum('ij,k->ikj', tempQ, s)

    gauge = gaugepi(Ks)

    gauss = cauchy(omega - tempE - tempQ, tol)



    greenp1 = green_pi_branch(Ks, pyp0)
    greenp2 = np.einsum('ijkl,a->ijakl', green_pi_branch(Qs, pyp0), np.ones(8))

    greenp2 = np.einsum('ijakl,iakl->ijakl', greenp2, gauge)

    inte = np.einsum('iakk, ibkjl->iab', greenp1, greenp2)

    inte = np.einsum('ijk, ijk->i', inte, gauss)

    return np.real(np.mean(inte))


def SSSF_pi(q, pyp0):
    Ks = pyp0.bigB
    Qs = Ks-q

    gauge = gaugepi(Ks)

    greenp1 = green_pi(Ks, pyp0)
    greenp2 = green_pi(Qs, pyp0)

    greenp2 = np.einsum('ikl,a->iakl',greenp2, np.ones(8))
    greenp2 = np.einsum('iakl,iakl->iakl', greenp2, gauge)

    inte = np.einsum('ikk, ikjl->i', greenp1, greenp2)

    return np.real(np.mean(inte))/4

def SSSF_pi_dumb(q, pyp0):
    Ks = pyp0.bigB
    Qs = Ks-q
    le = len(Ks)
    sQ = np.einsum('i,j->ij', np.ones(le), q)

    greenK = green_pi(Ks, pyp0)
    greenQ = green_pi(Qs, pyp0)

    greenp1 = greenK[:, 0:4, 0:4]
    greenp2 = greenQ[:, 0:4, 0:4]

    greenp1b = greenK[:, 4:8, 4:8]
    greenp2b = greenQ[:, 4:8, 4:8]

    temp = 0
    for rs in range(4):
        for i in range(4):
            for j in range(4):
                mu = unitCell(rs) + neta(0) * step(i)
                nu = unitCell(rs) + neta(0) * step(j)
                rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                index1 = findS(rs1)
                index2 = findS(rs2)
                temp += greenp1[:, rs, rs] * greenp2b[:, index1, index2]\
                        *np.exp(-1j * neta(0) * np.dot(Ks, NN(i)-NN(j)))\
                        *np.exp(1j * neta(0) * (A_pi(unitCell(rs), rs2) - A_pi(unitCell(rs), rs1)))
                temp += greenp1[:, index1, index2] * greenp2b[:, rs, rs]\
                        *np.exp(1j * neta(0) * np.dot(Ks, NN(i)-NN(j)))\
                        *np.exp(1j * neta(0) * (A_pi(unitCell(rs), rs2) - A_pi(unitCell(rs), rs1)))

                mu = unitCell(rs) + neta(1) * step(i)
                nu = unitCell(rs) + neta(1) * step(j)
                rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                index1 = findS(rs1)
                index2 = findS(rs2)
                temp += greenp1b[:, rs, rs] * greenp2[:, index1, index2]\
                        *np.exp(-1j * neta(1) * np.dot(Ks, NN(i)-NN(j)))\
                        *np.exp(1j * neta(1) * (A_pi(unitCell(rs), rs2) - A_pi(unitCell(rs), rs1)))

                temp += greenp1b[:, index1, index2] * greenp2b[:, index1, index2]\
                        *np.exp(1j * neta(1) * np.dot(Ks, NN(i)-NN(j)))\
                        *np.exp(1j * neta(1) * (A_pi(unitCell(rs), rs2) - A_pi(unitCell(rs), rs1)))

    return np.real(np.mean(temp))


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
            temp[i][j] = spinon_cont_zero(K[j], E[i], pyp0, tol)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            end = time.time()
            el = (end - start)*(totaltask-count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            sys.stdout.flush()
    return temp
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
        temp[j] = SSSF_zero(K[j], pyp0)
        # if temp[i][j] > tempMax:
        #     tempMax = temp[i][j]
        end = time.time()
        el = (end - start)*(totaltask-count)
        el = telltime(el)
        sys.stdout.write('\r')
        sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
        sys.stdout.flush()

    return temp
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
        temp[j] = SSSF_pi_dumb(K[j], pyp0)
        # if temp[i][j] > tempMax:
        #     tempMax = temp[i][j]
        end = time.time()
        el = (end - start)*(totaltask-count)
        el = telltime(el)
        sys.stdout.write('\r')
        sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
        sys.stdout.flush()
    return temp
    # E, K = np.meshgrid(e, K)





def telltime(sec):
    hours = math.floor(sec/3600)
    sec = sec-hours*3600
    minus = math.floor(sec/60)
    sec = int(sec - minus * 60)
    return str(hours) + ':' + str(minus) + ':' + str(sec)








