
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

def green_pi_old(k, alpha, pypi):
    E, V = pypi.LV_zero_old(k, alpha)
    V = np.conj(np.transpose(V, (0,2,1)))
    green = np.zeros((len(k),4,4), dtype=np.complex128)
    for i in range (4):
        for j in range (4):
            for k in range(4):
                green[:,i,j] += pypi.Jzz*V[:,j,k]*np.conj(V[:,i,k])/np.sqrt(2*pypi.Jzz*E[:,k])
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

def formfactorpm(K, q, alpha):
    M=np.zeros((len(K), 4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            M[:,i,j] = np.exp(-1j*neta(alpha)*np.dot(K, NN(i) -NN(j))) * np.exp(1j*neta(alpha)*np.dot(q, NN(i)- NN(j))/2)
    return M

def formfactormp(K, q, alpha):
    M=np.zeros((len(K), 4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            M[:,i,j] = np.exp(1j*neta(alpha)*np.dot(K, NN(i)- NN(j))) * np.exp(-1j*neta(alpha)*np.dot(q, NN(i)- NN(j))/2)
    return M

def formfactorpp(K, Q, q, alpha):
    M=np.zeros((len(K), 4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            M[:,i,j] = np.exp(-1j*neta(alpha)*np.dot(Q, NN(i))) * np.exp(-1j*neta(alpha)*np.dot(K, NN(j)))* np.exp(-1j*neta(alpha)*np.dot(q, NN(i)-NN(j))/2)
    return M

def formfactormm(K, Q, q, alpha):
    M=np.zeros((len(K), 4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            M[:,i,j] = np.exp(1j*neta(alpha)*np.dot(K, NN(i))) * np.exp(1j*neta(alpha)*np.dot(Q, NN(j)))* np.exp(-1j*neta(alpha)*np.dot(q, NN(i)-NN(j))/2)
    return M

def gaugefieldpipm(alpha, K, q):
    gauge = np.zeros((len(K), 4,4,4,4), dtype=complex)
    for rs in range(4):
        for rsp in range(4):
            for i in range(4):
                for j in range(4):
                    mu = unitCell(rs) + neta(alpha) * step(i)
                    nu = unitCell(rsp) + neta(alpha) * step(j)
                    rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                    rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                    index1 = findS(rs1)
                    index2 = findS(rs2)
                    gauge[:, rsp, rs, index1, index2] += np.exp(1j * neta(alpha) * (A_pi(unitCell(rs), rs1) - A_pi(unitCell(rsp), rs2))) \
                    * np.exp(1j * neta(alpha) * np.dot(K-q/2, NN(i)-NN(j)))
    return gauge/4

def gaugepipm(K, q):
    gauge = np.zeros((len(K), 8, 8, 8, 8), dtype=complex)
    gauge[:,0:4,0:4,4:8,4:8] = gaugefieldpipm(0, K, q)
    gauge[:,4:8,4:8, 0:4,0:4] = gaugefieldpipm(1, K, q)
    return gauge

def gaugefieldpimp(alpha, K, q):
    gauge = np.zeros((len(K), 4,4,4,4), dtype=complex)
    for rs in range(4):
        for rsp in range(4):
            for i in range(4):
                for j in range(4):
                    mu = unitCell(rs) + neta(alpha) * step(i)
                    nu = unitCell(rsp) + neta(alpha) * step(j)
                    rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                    rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                    index1 = findS(rs1)
                    index2 = findS(rs2)
                    gauge[:, rs, rsp, index2, index1] += np.exp(-1j * neta(alpha) * (A_pi(unitCell(rs), rs1) - A_pi(unitCell(rsp), rs2))) \
                    * np.exp(-1j * neta(alpha) * np.dot(K-q/2, NN(i)-NN(j)))
    return gauge/4
def gaugepimp(K, q):
    gauge = np.zeros((len(K), 8, 8, 8, 8), dtype=complex)
    gauge[:,0:4,0:4,4:8,4:8] = gaugefieldpipm(0, K, q)
    gauge[:,4:8,4:8,0:4,0:4] = gaugefieldpipm(1, K, q)
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


    ffacApm = formfactorpm(Qs, q, 0)
    ffacAmp = formfactormp(Ks, q, 0)
    ffacBpm = formfactorpm(Qs, q, 1)
    ffacBmp = formfactormp(Ks, q, 1)

    green = np.einsum('ijj,ikk->ijk',green_zero(Ks, pyp0), green_zero(Qs, pyp0))
    temp = np.multiply(cauchy(omega-tempE-tempQ, tol), green)
    temp = temp[:,0,1]*(ffacApm+ffacAmp)+ temp[:,1,0]*(ffacBpm+ffacBmp)

    return np.real(np.mean(temp))/2

def SSSF_zero(q, pyp0):
    Ks = pyp0.bigB
    Qs = Ks-q
    le = len(Ks)
    # sQ = np.einsum('i,j->ij', np.ones(le), q)

    greenp1 = green_zero(Ks, pyp0)
    greenp2 = green_zero(Qs, pyp0)

    #region S+- and S-+
    ffacApm = np.einsum('ijk->i',formfactorpm(Ks, q, 0))
    ffacAmp = np.einsum('ijk->i',formfactormp(Ks, q, 0))
    ffacBpm = np.einsum('ijk->i',formfactorpm(Ks, q, 1))
    ffacBmp = np.einsum('ijk->i',formfactormp(Ks, q, 1))

    greenA = np.einsum('ijj,ikk->ijk',greenp1, greenp2)
    greenpm = greenA[:,0,1]*(ffacApm+ffacAmp) + greenA[:,1,0]*(ffacBpm+ffacBmp)
    #endregion



    #region S++ and S--
    SppA = np.einsum('ijk->i',formfactorpp(Ks, Qs, q, 0))
    SppB = np.einsum('ijk->i',formfactorpp(Ks, Qs, q, 1))

    SmmA = np.einsum('ijk->i',formfactorpp(Ks, Qs, q, 0))
    SmmB = np.einsum('ijk->i',formfactorpp(Ks, Qs, q, 1))

    greenB = np.einsum('ijk,ikj->ijk', greenp1, greenp2)
    greenpp = greenB[:, 0, 1] * (SppA + SmmA) + greenB[:, 1, 0] * (SppB + SmmB)

    return np.real(np.mean(greenpm+greenpp)/4)/2


def spinon_cont_pi(q, omega, alpha, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks-q

    tempE = pyp0.E_pi(Ks, alpha, pyp0.lams)
    tempQ = pyp0.E_pi(Qs, alpha, pyp0.lams)

    s = np.array([1,1,1,1,1,1,1,1])
    tempE = np.einsum('ij,k->ijk', tempE, s)
    tempQ = np.einsum('ij,k->ikj', tempQ, s)

    gauge = gaugepipm(Ks,q)

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

    gaugepm = gaugepipm(Ks, q)
    gaugemp = gaugepimp(Ks, q)

    greenpK = green_pi(Ks, pyp0)
    greenpQ = green_pi(Qs, pyp0)

    greenp1 = np.einsum('ikl,a,b->iabkl', greenpQ, np.ones(8), np.ones(8))
    greenp1 = np.einsum('iabkl,iabkl->iabkl', greenp1, gaugepm)

    inte = np.einsum('ijk, ijkla->i', greenpK, greenp1)

    # greenp2 = np.einsum('ikl,a,b->iabkl', greenpQ, np.ones(8), np.ones(8))
    # greenp2 = np.einsum('iabkl,iabkl->iabkl', greenp2, gaugepm)
    #
    # inte += np.einsum('ijk, ijkla->i', greenpK, greenp2)

    return np.real(np.mean(inte))/4

def SSSF_pi_dumb(q, pyp0):
    Ks = pyp0.bigB
    Qs = Ks-q
    le = len(Ks)


    # greenK = green_pi(Ks, pyp0)
    # greenQ = green_pi(Qs, pyp0)



    greenpKA = green_pi_old(Ks, 0, pyp0)
    greenpKB = green_pi_old(Ks, 1, pyp0)
    greenpQA = green_pi_old(Qs, 0, pyp0)
    greenpQB = green_pi_old(Qs, 1, pyp0)


    temp1 = np.zeros(le, dtype=np.complex128)
    temp2 = np.zeros(le, dtype=np.complex128)
    for rs in range(4):
        for rsp in range(4):
            for i in range(4):
                for j in range(4):
                    mu = unitCell(rs) + step(i)
                    nu = unitCell(rsp) + step(j)
                    rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
                    rs2 = np.array([nu[0] % 1, nu[1] % 2, nu[2] % 2])
                    index1 = findS(rs1)
                    index2 = findS(rs2)

                    temp1 += greenpKA[:, rs, rsp] * greenpQB[:, index2, index1]\
                            *np.exp(1j * np.dot(Ks-q/2, NN(i)-NN(j))) \
                            *np.exp(1j * (A_pi(unitCell(rs), rs1) - A_pi(unitCell(rsp), rs2)))/4

                    temp2 += greenpQA[:, rsp, rs] * greenpKB[:, index1, index2]\
                            *np.exp(-1j * np.dot(Ks-q/2, NN(i) -NN(j))) \
                            *np.exp(-1j * (A_pi(unitCell(rs), rs1) - A_pi(unitCell(rsp), rs2)))/4


    return [np.real(np.mean(temp1)),np.real(np.mean(temp2))]


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
    totaltask = K.shape[0]*K.shape[1]
    increment = totaltask/50
    count = 0
    temp = np.zeros((K.shape[0], K.shape[1]))
    temp1 = np.zeros((K.shape[0], K.shape[1]))

    for i in range(len(K)):
        for j in range(K.shape[1]):
            start = time.time()
            count = count + 1
            temp[i,j], temp1[i,j] = SSSF_pi_dumb(K[i,j], pyp0)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            end = time.time()
            el = (end - start)*(totaltask-count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            sys.stdout.flush()
    return [temp, temp1]
    # E, K = np.meshgrid(e, K)





def telltime(sec):
    hours = math.floor(sec/3600)
    sec = sec-hours*3600
    minus = math.floor(sec/60)
    sec = int(sec - minus * 60)
    return str(hours) + ':' + str(minus) + ':' + str(sec)








