import numpy as np
import math
from misc_helper import *
# Gamma = np.array([0, 0, 0])
# L = np.pi * np.array([1, 1, 1])
# X = 2 * np.pi * np.array([0, 1, 0])
# W = 2 * np.pi * np.array([0, 1, 1 / 2])
# K = 2 * np.pi * np.array([0, 3 / 4, 3 / 4])
# U = 2 * np.pi * np.array([1 / 4, 1, 1 / 4])
#
# GammaX = np.linspace(Gamma, X, 20)
#
# print(np.einsum('ij,j->i', GammaX, L))


# H = np.linspace(-2, 2, 5)
# L = np.linspace(-2, 2, 5)
#
# X,Y = np.meshgrid(H, L)
#
# print(X)
# print(Y)

# arr = np.array([-10.2, -5.3, -2.1, 0, 1.2, 3.4])
# print(np.diag(arr))

# arr = np.array([[1, 2j], [-2j, 1]])
# E, V = np.linalg.eigh(arr)
# print(E,V)
# print(arr*V[:,0])
# print(arr*V[:,1])

for i in range(4):
    for j in range(4):
        mu = unitCell(i) + step(j)
        rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
        index1 = findS(rs1)
        print("rs is " + str(i) + " step is " + str(j))
        print(index1)
        index1 = np.array(np.where(piunitcell[j,i]==1))[0,0]
        print(index1)
        print('-------------------------------------')

