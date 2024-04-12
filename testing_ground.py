import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sp
from sympy import re, im, I, E, symbols
from misc_helper import *
from sympy.printing.mathml import mathml
sp.init_printing(use_unicode=True) # allow LaTeX printing

# from misc_helper import *
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

# for i in range(4):
#     for j in range(4):
#         mu = unitCell(i) + step(j)
#         rs1 = np.array([mu[0] % 1, mu[1] % 2, mu[2] % 2])
#         index1 = findS(rs1)
#         print("rs is " + str(i) + " step is " + str(j))
#         print(index1)
#         index1 = np.array(np.where(piunitcell[j,i]==1))[0,0]
#         print(index1)
#         print('-------------------------------------')
#
def make_dashedLines(x,y,z,ax):
    for i in range(0, len(x)):
        x_val, y_val, z_val = x[i],y[i],z[i]
        ax.plot([0,x_val],[y_val,y_val],zs=[0,0], linestyle="dashed",color="black")
        ax.plot([x_val,x_val],[0,y_val],zs=[0,0], linestyle="dashed",color="black")
        ax.plot([x_val,x_val],[y_val,y_val],zs=[0,z_val], linestyle="dashed",color="black")

def Api(A, mu):
    A1, A2, A3 = A
    if mu == 0:
        return 0
    elif mu == 1:
        return np.pi*(A2+A3)
    elif mu == 2:
        return np.pi*A3
    else:
        return 0

def Api_sungbin(A, mu):
    emu = np.array([0,1,1,0])
    return emu[mu] * np.dot(A, 2*np.pi*np.array([1,0,0]))

def color(A, mu):
    temp = np.real(np.exp(1j*Api(A,mu)))
    if temp == 1:
        return 'k'
    else:
        return 'r'


def color_sungbin(A, mu):
    temp = np.real(np.exp(1j*Api_sungbin(A,mu)))
    if temp == 1:
        return 'k'
    else:
        return 'r'

fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
e = [[0,1/2,1/2], [1/2, 0, 1/2], [1/2 ,1/2 , 0]]

def drawline(A,B, c):
    draw = np.linspace(A,B,2)
    ax.plot(draw[:,0], draw[:,1], draw[:,2], c=c)

def cartesian(A, alpha):
    return np.einsum('ij, jk->ik',A, e) + alpha * np.array([1/4,1/4,1/4])/2

unitcell = [[0,0,0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
unitcell_sungbin = [[0,0,0],  [0, 0, 1]]
NN = np.array([np.array([-1 / 4, -1 / 4, -1 / 4]), np.array([-1 / 4, 1 / 4, 1 / 4]), np.array([1 / 4, -1 / 4, 1 / 4]), np.array([1 / 4, 1 / 4, -1 / 4])])
step = [[0,0,0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

def graphgauge_DOWN():
    rUp = []

    for i in range(4):
        for j in range(4):
            rUp += [np.array(unitcell[i]) + np.array(step[j])]

    rUp = np.array(rUp)
    rDown = np.array(unitcell)

    rUpcart = cartesian(rUp, -1)
    rDowncart = cartesian(rDown, 1)

    ax.scatter(rDowncart[:,0], rDowncart[:,1], rDowncart[:,2], c='b',s=50)
    ax.scatter(rUpcart[:,0], rUpcart[:,1], rUpcart[:,2], c='g',s=50)

    for i in range(len(rDown)):
        curr = rDowncart[i]
        drawline(curr, curr+NN[0], color(rDown[i], 0))
        drawline(curr, curr+NN[1], color(rDown[i], 1))
        drawline(curr, curr+NN[2], color(rDown[i], 2))
        drawline(curr, curr+NN[3], color(rDown[i], 3))


    # ax.scatter( x,y,z, c='r', marker='o')
    # make_dashedLines(x,y,z,ax)

    # Make a 3D quiver plot
    # x, y, z = np.array([[0,0,0],[0,0,0],[0,0,0]])
    # u, v, w = np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]])
    # ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    # ax.grid(False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def graphgauge_UP():
    rDown = []

    for i in range(4):
        for j in range(4):
            rDown += [np.array(unitcell[i]) - np.array(step[j])]


    rDown = np.array(rDown)
    rUp = np.array(unitcell)

    rUpcart = cartesian(rUp, -1)
    rDowncart = cartesian(rDown, 1)

    ax.scatter(rDowncart[:, 0], rDowncart[:, 1], rDowncart[:, 2], c='b', s=50)
    ax.scatter(rUpcart[:, 0], rUpcart[:, 1], rUpcart[:, 2], c='g', s=50)

    for i in range(len(rUp)):
        curr = rUpcart[i]
        drawline(curr, curr - NN[0], color(rUp[i]-step[0], 0))
        drawline(curr, curr - NN[1], color(rUp[i]-step[1], 1))
        drawline(curr, curr - NN[2], color(rUp[i]-step[2], 2))
        drawline(curr, curr - NN[3], color(rUp[i]-step[3], 3))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def graphgauge_DOWN_sungbin():
    rUp = []

    for i in range(2):
        for j in range(4):
            rUp += [np.array(unitcell_sungbin[i]) + np.array(step[j])]

    rUp = np.array(rUp)
    rDown = np.array(unitcell_sungbin)

    rUpcart = cartesian(rUp, 0)
    rDowncart = cartesian(rDown, 2)

    ax.scatter(rDowncart[:,0], rDowncart[:,1], rDowncart[:,2], c='b',s=50)
    ax.scatter(rUpcart[:,0], rUpcart[:,1], rUpcart[:,2], c='g',s=50)

    for i in range(len(rDown)):
        curr = rDowncart[i]
        drawline(curr, curr+NN[0], color_sungbin(rDowncart[i]+NN[0], 0))
        drawline(curr, curr+NN[1], color_sungbin(rDowncart[i]+NN[1], 1))
        drawline(curr, curr+NN[2], color_sungbin(rDowncart[i]+NN[2], 2))
        drawline(curr, curr+NN[3], color_sungbin(rDowncart[i]+NN[3], 3))


    # ax.scatter( x,y,z, c='r', marker='o')
    # make_dashedLines(x,y,z,ax)

    # Make a 3D quiver plot
    # x, y, z = np.array([[0,0,0],[0,0,0],[0,0,0]])
    # u, v, w = np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]])
    # ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    # ax.grid(False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def graphgauge_UP_sungbin():
    rDown = []

    for i in range(2):
        for j in range(4):
            rDown += [np.array(unitcell_sungbin[i]) - np.array(step[j])]


    rDown = np.array(rDown)
    rUp = np.array(unitcell_sungbin)

    rUpcart = cartesian(rUp, 0)
    rDowncart = cartesian(rDown, 2)

    ax.scatter(rDowncart[:, 0], rDowncart[:, 1], rDowncart[:, 2], c='b', s=50)
    ax.scatter(rUpcart[:, 0], rUpcart[:, 1], rUpcart[:, 2], c='g', s=50)

    for i in range(len(rUp)):
        curr = rUpcart[i]
        drawline(curr, curr - NN[0], color_sungbin(rUpcart[i], 0))
        drawline(curr, curr - NN[1], color_sungbin(rUpcart[i], 1))
        drawline(curr, curr - NN[2], color_sungbin(rUpcart[i], 2))
        drawline(curr, curr - NN[3], color_sungbin(rUpcart[i], 3))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


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

NNs = sp.Matrix([[-1 / 4, -1 / 4, -1 / 4], [-1 / 4, 1 / 4, 1 / 4], [1 / 4, -1 / 4, 1 / 4], [1 / 4, 1 / 4, -1 / 4]])

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

def algHamSungbin():
    M_down = sp.Matrix([[0,0],[0,0]])
    M_up = sp.Matrix([[0,0],[0,0]])
    kx, ky, kz = sp.symbols("kx ky kz")
    KK = sp.Matrix([kx,ky,kz])
    for k in range(2):
        for i in range(4):
            for j in range(4):
                if not i == j:
                    index1 = np.array(np.where(piunitcell_here[i, k] == 1))[0,0]
                    index2 = np.array(np.where(piunitcell_here[j, k] == 1))[0,0]
                    M_down[index1, index2] += sp.exp(I* KK.dot(NN[i] - NN[j])) * A_pi_rs_traced_here_down[k, i, j]
                    M_up[index1, index2] += sp.exp(-I * KK.dot(NN[i] - NN[j])) * A_pi_rs_traced_here_up[k, i, j]

    sp.print_latex(M_up)

# graphgauge_UP()
# graphgauge_DOWN()

# graphgauge_UP_sungbin()
# graphgauge_DOWN_sungbin()

# plt.show()

# algHamSungbin()

# lams = -np.pi/2
#
# A = np.zeros((3,4))
#
# print(np.mean(A, axis=0))

# n1 = 5
# n2 = 10
# JH = np.mgrid[0:3:1j*n1, 0:100:1j*n2].reshape(2,-1).T
# print(JH.shape)
# A = np.linspace(0, 3, n1)
# B = np.linspace(0, 100, n2)
# X, Y = np.meshgrid(A, B)
# test = np.zeros(n1*n2)
# for i in range(n1*n2):
#     test[i] = JH[i,0]+JH[i,1]
# print(test)
# test = test.reshape((n1, n2))
# plt.pcolormesh(X, Y, test.T)
# plt.colorbar()
# plt.show()

# a = np.array([[1,2,3],[4,5,6]])

# a1 = np.array([[0,0,0],
#               [1,0,0],
#               [0,1,0],
#               [0,0,1],
#               [1,1,1]])
# a2 = np.array([[0,0,0],
#                [1,1,1]])
#
# a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
# print(a1_rows)
# a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
# print(a1_rows)
# print(np.in1d(a1_rows, a2_rows))
# print(np.where(np.in1d(a1_rows, a2_rows)==False)[0])
# print(np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1]))
#
#

# def f(k, a, b, c):
#     x,y,z=k
#     return a*x**2+b*y**2+c*z**2
#
# n = 10
#
# print(adaptive_gauss_quadrature_3d(f, -1, 1, -1, 1, -1, 1, 1e-16,1,1,1))

# A = np.array([1,1,1])
# B = np.arange(48).reshape((3,2,2,4))
# C = np.einsum('ijkl->jkil', B)
# print(np.dot(A,C))
# print(np.einsum('i, ijkl->jkl', A, B))
import numpy as np
import numba as nb


@nb.njit
def equivalence_relation(x, y):
    # Define your equivalence relation here
    # For example, let's say two elements are equivalent if their sum is the same
    return np.sum(x) == np.sum(y)


@nb.njit(parallel=True)
def find_quotient_group(arr, equiv_relation):
    # Step 1: Initialize an empty list to store equivalence classes
    equiv_classes = []

    # Step 2: Find equivalence classes
    for i in range(arr.shape[0]):
        x = arr[i]
        added = False
        for equiv_class in equiv_classes:
            if equiv_relation(x, equiv_class[0]):
                equiv_class.append(x)
                added = True
                break
        if not added:
            equiv_classes.append([x])
    representatives = np.zeros((len(equiv_classes),3),dtype=np.float64)
    for i in range(len(equiv_classes)):
        for j in range(3):
            representatives[i,j]=equiv_classes[i][0][j]
    return representatives


# Example usage
# arr = np.array([[0,1,1],[0,0,1],[1,0,1]])
# quotient_group = find_quotient_group(arr, equi_class_110)
# print("Quotient Group:")
# print(quotient_group)
# test= np.array(quotient_group, dtype=object)
# print(test.shape)
# print(test[0])
from matplotlib.colors import LogNorm
#
# dx, dy = 0.015, 0.05
# y, x = np.mgrid[slice(-4, 4 + dy, dy),
# slice(-4, 4 + dx, dx)]
# z = (1 - x / 3. + x ** 5 + y ** 5) * np.exp(-x ** 2 - y ** 2)
# z = z[:-1, :-1]
# z_min, z_max = -np.abs(z).max(), np.abs(z).max()
#
# c = plt.imshow(z, cmap='Greens', vmin=z_min, vmax=z_max,
#                extent=[x.min(), x.max(), 3, -3],
#                interpolation='nearest', origin='lower', aspect='auto')
# plt.colorbar(c)
#
# plt.title('matplotlib.pyplot.imshow() function Example',
#           fontweight="bold")
# plt.show()

a, b, c = 1, 1, 0
print(a*np.array([0,0.5,0.5])+b*np.array([0.5,0,0.5])+c*np.array([0.5,0.5,0]))

a, b, c = 2, 2, 0
print(a*np.array([0,0.5,0.5])+b*np.array([0.5,0,0.5])+c*np.array([0.5,0.5,0]))

a, b, c = 2, 2, 0.5
print(a*np.array([0,0.5,0.5])+b*np.array([0.5,0,0.5])+c*np.array([0.5,0.5,0]))

a, b, c = 2, 2, 1
print(a*np.array([0,0.5,0.5])+b*np.array([0.5,0,0.5])+c*np.array([0.5,0.5,0]))

a, b, c = 2, 2, 2
print(a*np.array([0,0.5,0.5])+b*np.array([0.5,0,0.5])+c*np.array([0.5,0.5,0]))

a, b, c = 1, 1, 1
print(a*np.array([0,0.5,0.5])+b*np.array([0.5,0,0.5])+c*np.array([0.5,0.5,0]))

a, b, c = 0.5, 0.5, 0.5
print(a*np.array([0,0.5,0.5])+b*np.array([0.5,0,0.5])+c*np.array([0.5,0.5,0]))