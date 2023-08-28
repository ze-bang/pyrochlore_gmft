
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
    Vt = contract('ijk,ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    green = pyp0.Jzz/np.sqrt(2*pyp0.Jzz*E)
    green = contract('ikjl, ik->ijl', Vt, green)
    return green

def green_zero_branch(k, pyp0):
    E, V = pyp0.LV_zero(k)
    Vt = contract('ijk,ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    green = pyp0.Jzz/np.sqrt(2*pyp0.Jzz*E)
    green = contract('ikjl, ik->ikjl', Vt, green)
    return green

def green_pi(k, pypi):
    E, V = pypi.LV_zero(k)
    Vt = contract('ijk, ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    # temp = 2*pypi.Jzz*np.multiply(pypi.V[:,nu,i], np.conj(np.transpose(pypi.V, (0, 2, 1)))[:,i,mu])
    green = pypi.Jzz/np.sqrt(2*pypi.Jzz*E)
    green = contract('ikjl, ik->ijl', Vt, green)
    return green

def green_pi_old(k, alpha, pypi):
    E, V = pypi.LV_zero_old(k, alpha)
    Vt = contract('ijk, ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    # temp = 2*pypi.Jzz*np.multiply(pypi.V[:,nu,i], np.conj(np.transpose(pypi.V, (0, 2, 1)))[:,i,mu])
    green = pypi.Jzz/np.sqrt(2*pypi.Jzz*E)
    green = contract('ikjl, ik->ijl', Vt, green)
    return green

def green_pi_branch(k, pypi):
    E, V = pypi.LV_zero(k)
    Vt = contract('ijk, ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    # temp = 2*pypi.Jzz*np.multiply(pypi.V[:,nu,i], np.conj(np.transpose(pypi.V, (0, 2, 1)))[:,i,mu])
    green = pypi.Jzz/np.sqrt(2*pypi.Jzz*E)
    green = contract('ikjl, ik->ikjl', Vt, green)
    return green

def green_pi_old_branch(k, alpha, pypi):
    E, V = pypi.LV_zero_old(k, alpha)
    Vt = contract('ijk, ikl->iklj', V, np.transpose(np.conj(V), (0,2,1)))
    # temp = 2*pypi.Jzz*np.multiply(pypi.V[:,nu,i], np.conj(np.transpose(pypi.V, (0, 2, 1)))[:,i,mu])
    green = pypi.Jzz/np.sqrt(2*pypi.Jzz*E)
    green = contract('ikjl, ik->ikjl', Vt, green)
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

def bose(beta, omega):
    if beta == 0:
        return np.zeros(omega.shape)
    else:
        return 1/(np.exp(beta*omega)-1)

def delta(Ek, Eq, omega, tol):
    beta = 0
    size = Ek.shape[1]
    Ekenlarged = contract('ik,j->ikj',Ek, np.ones(size))
    Eqenlarged = contract('ik,j->ijk', Eq, np.ones(size))
    A = contract('ia, ib, iab->iab' ,1+bose(beta, Ek) ,1+bose(beta, Eq), cauchy(omega - Ekenlarged - Eqenlarged, tol))
    B = contract('ia, ib, iab->iab' ,1+bose(beta, Ek) ,bose(beta, Eq), cauchy(omega - Ekenlarged + Eqenlarged, tol))
    C = contract('ia, ib, iab->iab' ,bose(beta, Ek) ,1+bose(beta, Eq), cauchy(omega + Ekenlarged - Eqenlarged, tol))
    D = contract('ia, ib, iab->iab', bose(beta, Ek), bose(beta, Eq), cauchy(omega + Ekenlarged + Eqenlarged, tol))
    return A+B+C+D



def g(q):
    M = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            M[i,j] = np.dot(z[i], z[j]) - np.dot(z[i],q) * np.dot(z[j],q)/ np.dot(q,q)
    return M

def gNSF(v):
    M = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            M[i,j] = np.dot(z[i], v) * np.dot(z[j], v)
    return M

def DSSF_zero(q, omega, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks-q

    tempE= pyp0.E_zero(Ks)
    tempQ = pyp0.E_zero(Qs)


    greenp1 = green_zero_branch(Ks, pyp0)
    greenp2 = green_zero_branch(Qs, pyp0)

    #region S+- and S-+
    deltapm = delta(tempE, tempQ, omega,tol)
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)

    greenA = contract('ia, ib, iab->i',greenp1[:, :, 0,0], greenp2[:,:, 1,1], deltapm)
    greenpm = contract('i,ijk->ijk',greenA, (ffactpm+np.conj(np.transpose(ffactpm, (0,2,1)))))/4
    #endregion

    #region S++ and S--
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)

    greenB = contract('ia, ib, iab->i',greenp1[:, :, 0,1], greenp2[:,:, 1,0], deltapm)
    greenpp = contract('i,ijk->ijk',greenB, (ffactpp+np.conj(np.transpose(ffactpp, (0,2,1)))))/4

    S = (greenpp + greenpm)/4
    Sglobal = contract('ijk,jk->i', S, g(q))
    S = contract('ijk->i',S)

    return np.real(np.mean(S)), np.real(np.mean(Sglobal))

def SSSF_zero(q, v, pyp0):
    Ks = pyp0.bigB
    Qs = Ks-q
    le = len(Ks)
    # sQ = contract('i,j->ij', np.ones(le), q)

    greenp1 = green_zero(Ks, pyp0)
    greenp2 = green_zero(Qs, pyp0)

    #region S+- and S-+
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)

    greenA = greenp1[:,0,0] * greenp2[:,1,1]
    greenpm = contract('i,ijk->ijk',greenA, (ffactpm+np.conj(np.transpose(ffactpm, (0,2,1)))))/4
    #endregion

    #region S++ and S--
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNplus)
    ffactpp = np.exp(1j * ffact)

    greenB = greenp1[:,0,1] * greenp2[:,1,0]
    greenpp = contract('i,ijk->ijk',greenB, (ffactpp+np.conj(np.transpose(ffactpp, (0,2,1)))))/4

    S = (greenpp + greenpm)/4
    Sglobal = contract('ijk,jk->i',S, g(q))
    SNSF = contract('ijk,jk->i',S, gNSF(v))
    S = contract('ijk->i',S)

    return np.real(np.mean(S)), np.real(np.mean(Sglobal)), np.real(np.mean(SNSF))

def DSSF_pi(q, omega, pyp0, tol):
    Ks = pyp0.bigB
    Qs = Ks - q

    tempE = pyp0.LV_zero_old(Ks, 0)[0]
    tempQ = pyp0.LV_zero_old(Qs, 0)[0]

    greenpKA = green_pi_old_branch(Ks, 0, pyp0)
    greenpKB = np.conj(greenpKA)
    greenpQA = green_pi_old_branch(Qs, 0, pyp0)
    greenpQB = np.conj(greenpQA)

    deltapm = delta(tempE, tempQ, omega, tol)
    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)


    Spm = contract('ioab, ipyx, iop, abjk, jax, kby, ijk->ijk', greenpKA, greenpQB, deltapm, A_pi_rs_rsp, piunitcell, piunitcell,
                    ffactpm)/4

    Smp = contract('ipba, ioxy, iop, abjk, jax, kby, ijk->ijk', greenpQA, greenpKB, deltapm, A_pi_rs_rsp, piunitcell, piunitcell,
                    np.conj(ffactpm))/4

    S = (Spm + Smp)/4

    Sglobal = contract('ijk,jk->i', S, g(q))
    S = contract('ijk->i',S)


    return np.real(np.mean(S)), np.real(np.mean(Sglobal))


def SSSF_pi(q, v, pyp0):
    Ks = pyp0.bigB
    Qs = Ks - q
    v = v / magnitude(v)
    le = len(Ks)

    greenpKA = green_pi_old(Ks, 0, pyp0)
    greenpKB = np.conj(greenpKA)
    greenpQA = green_pi_old(Qs, 0, pyp0)
    greenpQB = np.conj(greenpQA)

    ffact = contract('ik, jlk->ijl', Ks - q / 2, NNminus)
    ffactpm = np.exp(1j * ffact)

    Spm = contract('iab, iyx, abjk, jax, kby, ijk->ijk', greenpKA, greenpQB, A_pi_rs_rsp, piunitcell, piunitcell,
                    ffactpm)/4
    Smp = contract('iba, ixy, abjk, jax, kby, ijk->ijk', greenpQA, greenpKB, A_pi_rs_rsp, piunitcell, piunitcell,
                    np.conj(ffactpm))/4

    S = (Spm + Smp)/4

    Sglobal = contract('ijk,jk->i',S, g(q))
    SNSF = contract('ijk,jk->i',S, gNSF(v))
    S = contract('ijk->i',S)

    return np.real(np.mean(S)), np.real(np.mean(Sglobal)), np.real(np.mean(SNSF))

def SSSF_pi_dumb(q, v, pyp0):
    Ks = pyp0.bigB
    Qs = Ks-q
    v = v/magnitude(v)
    le = len(Ks)

    greenpKA = green_pi_old(Ks, 0, pyp0)
    greenpKB = np.conj(greenpKA)
    greenpQA = green_pi_old(Qs, 0, pyp0)
    greenpQB = np.conj(greenpQA)


    temp = np.zeros(le, dtype=np.complex128)
    temp1 = np.zeros(le, dtype=np.complex128)
    temp2 = np.zeros(le, dtype=np.complex128)
    for rs in range(4):
        for rsp in range(4):
            for i in range(4):
                for j in range(4):
                    index1 = np.array(np.where(piunitcell[i, rs] == 1))[0, 0]
                    index2 = np.array(np.where(piunitcell[j, rsp] == 1))[0, 0]

                    Spm = greenpKA[:, rs, rsp] * greenpQB[:, index2, index1]\
                            *np.exp(1j * np.dot(Ks-q/2, NN[i]-NN[j])) \
                            *np.exp(1j * (A_pi[rs, i] - A_pi[rsp, j]))/4
                    Smp = greenpQA[:, rsp, rs] * greenpKB[:, index1, index2]\
                            *np.exp(-1j * np.dot(Ks-q/2, NN[i]-NN[j])) \
                            *np.exp(-1j * (A_pi[rs, i] - A_pi[rsp, j]))/4

                    temp += (Spm + Smp)/4
                    temp1 += (Spm + Smp) * (np.dot(z[i], z[j]) - np.dot(z[i], q) * np.dot(z[j], q) / np.dot(q, q))/4
                    temp2 += (Spm + Smp) * (np.dot(z[i], v) * np.dot(z[j], v))/4


    return np.real(np.mean(temp)), np.real(np.mean(temp1)), np.real(np.mean(temp2))

def graph_DSSF_pi(pyp0, E, K, tol):
    el = "==:==:=="
    totaltask = len(E)*len(K)
    increment = totaltask/50
    count = 0

    temp = np.zeros((len(E), len(K)))
    temp1 = np.zeros((len(E), len(K)))

    for i in range(len(E)):
        for j in range(len(K)):
            start = time.time()
            count = count + 1
            temp[i,j], temp1[i,j] = DSSF_pi(K[j], E[i], pyp0, tol)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            end = time.time()
            el = (end - start)*(totaltask-count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            sys.stdout.flush()
    return temp, temp1


def graph_DSSF_zero(pyp0, E, K, tol):
    el = "==:==:=="
    totaltask = len(E)*len(K)
    increment = totaltask/50
    count = 0
    temp = np.zeros((len(E), len(K)))
    temp1 = np.zeros((len(E), len(K)))
    for i in range(len(E)):
        for j in range(len(K)):
            start = time.time()
            count = count + 1
            temp[i,j], temp1[i,j] = DSSF_zero(K[j], E[i], pyp0, tol)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            end = time.time()
            el = (end - start)*(totaltask-count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            sys.stdout.flush()
    return temp, temp1


def graph_SSSF_zero(pyp0, K, V):
    el = "==:==:=="
    totaltask =  K.shape[0]*K.shape[1]
    increment = totaltask/50
    count = 0
    temp = np.zeros((K.shape[0], K.shape[1]))
    temp1 = np.zeros((K.shape[0], K.shape[1]))
    temp2 = np.zeros((K.shape[0], K.shape[1]))

    for i in range(len(K)):
        for j in range(K.shape[1]):
            start = time.time()
            count = count + 1
            temp[i,j], temp1[i,j], temp2[i,j] = SSSF_zero(K[i,j],V, pyp0)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            end = time.time()
            el = (end - start)*(totaltask-count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            sys.stdout.flush()
    return temp, temp1, temp2
    # E, K = np.meshgrid(e, K)


def graph_SSSF_pi(pyp0, K, V):
    el = "==:==:=="
    totaltask = K.shape[0]*K.shape[1]
    increment = totaltask/50
    count = 0
    temp = np.zeros((K.shape[0], K.shape[1]))
    temp1 = np.zeros((K.shape[0], K.shape[1]))
    temp2 = np.zeros((K.shape[0], K.shape[1]))
    for i in range(len(K)):
        for j in range(K.shape[1]):
            start = time.time()
            count = count + 1
            temp[i,j], temp1[i,j], temp2[i,j] = SSSF_pi(K[i,j],V, pyp0)
            # if temp[i][j] > tempMax:
            #     tempMax = temp[i][j]
            end = time.time()
            el = (end - start)*(totaltask-count)
            el = telltime(el)
            sys.stdout.write('\r')
            sys.stdout.write("[%s] %f%% Estimated Time: %s" % ('=' * int(count/increment) + '-'*(50-int(count/increment)), count/totaltask*100, el))
            sys.stdout.flush()
    return temp, temp1, temp2
    # E, K = np.meshgrid(e, K)





def telltime(sec):
    hours = math.floor(sec/3600)
    sec = sec-hours*3600
    minus = math.floor(sec/60)
    sec = int(sec - minus * 60)
    return str(hours) + ':' + str(minus) + ':' + str(sec)








