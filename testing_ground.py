import numpy as np
import math

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


H = np.linspace(-2, 2, 5)
L = np.linspace(-2, 2, 5)

X,Y = np.meshgrid(H, L)

print(X)
print(Y)