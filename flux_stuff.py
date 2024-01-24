import numpy as np
from opt_einsum import contract
def indexRule(K, unitCell):
    K1, K2, K3 = K
    return np.array([0, K2%2, K3%2])
def getIndex(K, unitCell):
    for i in range(4):
        if (K == unitCell[i]).all():
            return i
def calFlux(site, A_pi, unitCell):
    # pyrochlore's dual lattice is FCC so each tetrahedron has 12 nearest neighbour. In the format of [(e1,e2,e3), unitcell index]
    # [(1, 0, 0), 0], [(0, 1, 0), 1], [(0, 0, 1), 2]
    # [-(1, 0, 0), 0], [-(0, 1, 0), 1], [-(0, 0, 1), 2]
    # [(1, -1, 0), 1], [(1, 0, -1), 2], [(0, 1, -1), 3]
    # [(-1, 1, 0), 1], [(-1, 0, 1), 2], [(0, -1, 1), 3]

    # Like in Alaric's result, the unit flux are 012, 123, 230, 301,

    # A tetrahedron has 6 sides. Each sides will affect 2 flux rings. For example, the 01 side will be accounted in 012 and 301
    # Therefore, each tetrahedron will have 12 affect flux rings. In the case, where our unit cell is defined like a diamond
    # there are in total,

    # Let us think about this in terms of the kagome lattice first, in Alaric's result, they claim that pi pi 0 0 flux has
    # pi flux on 012 and 123, in another word, if we look from the perpendicular angle to these flux rings (looking at a kagome)
    # then all of the flux rings on that surface will be pi flux.

    # So let us think about the first question, how does a pi flux kagome look like in the first place.
    # First of all, consider a surface with only one type of loop. For example, 012
    # Notice that you need at least unit cell of 2 in order to make it pi flux. Since a hex ring will repeat each unit side twice,
    # if you have only unit cell of one then it will simply be accounted for twice, resulting in 0 flux.

    # The easiest pi flux unit cell is simply one cell with only one side pi flux and the other completely 0 flux

    # Now let's consider it for a unit cell of four tetrahedron. First of all, what are the relevant flux rings for a single tetrahedron
    # Let us compute this base case and then apply this for the four.

    # Starting from the tetrahedron, we have 6 sides, let us label it with the same notation as in alaric's
    # 01, 02, 03, 12, 13, 23
    # First of all, let us realize to make a hex ring we require 3 tetrahedrons
    # For 01, two flux rings are possible, 012, and 301.
        # If we do 013, then the three tetrahedron required (first one being the starting one) labeled by (e1, e2, e3)
        # are (0,0,0), (1, 0, -1), (0, 0, -1)
        # To see a pattern. Notice 01 has vector direction (0,1,1). and since we choose the next bond to be 13 which has vector (-1, 0, 1)
        # Therefore, the hex ring must be perpendicular to the surface spammed by these two vectors. That is, it must be perpendicular to
        # (0, 1, 1) x (-1, 0, 1) = (1, 1, 1)
    # Know that we know the tetrahedron involved, we consider the exact sites from the tetrahedron:
    # [(0,0,0), 0, 1], [(-1, 0, 1), 3, 0], [(0, 0, -1), 1, 3] But this is easy since we know that it must follow from 013 cyclic order. Let us do this for
    # All sides and flux rings:
    # 01:
        #012: [(0,0,0), 0, 1], [(1, -1, 0), 2, 0], [(0, -1, 0), 1, 2] A
        #013: [(0,0,0), 0, 1], [(1, 0, -1), 3, 0], [(0, 0, -1), 1, 3] D
    # 02:
        #021: [(0,0,0), 0, 2], [(-1, 1, 0), 1, 0], [(-1, 0, 0), 2, 1] A
        #023: [(0,0,0), 0, 2], [(0, 1, -1), 3, 0], [(0, 0, -1), 2, 3] C
    # 03:
        #031: [(0,0,0), 0, 3], [(-1, 0, 1), 1, 0], [(-1, 0, 0), 3, 1] D
        #032: [(0,0,0), 0, 3], [(0, -1, 1), 2, 0], [(0, -1, 0), 3, 2] C
    # 12:
        #120: [(0,0,0), 1, 2], [(0, 1, 0), 0, 1], [(1, 0, 0), 2, 0] A
        #123: [(0,0,0), 1, 2], [(0, 1, -1), 3, 1], [(1, 0, -1), 2, 3] B
    # 13:
        #130: [(0,0,0), 1, 3], [(0, 0, 1), 0, 1], [(1, 0, 0), 3, 0] D
        #132: [(0,0,0), 1, 3], [(0, -1, 1), 2, 1], [(1, -1, 0), 3, 2] B
    # 23:
        #230: [(0,0,0), 2, 3], [(0, 1, 0), 0, 2], [(0, 0, 1), 3, 0] C
        #231: [(0,0,0), 2, 3], [(-1, 1, 0), 1, 2], [(-1, 0, 1), 3, 1] B

    # Let us label all 4 types of ring as 012 = A, 123 = B, 230 = C, 301 = D

    # Now let us put this into code organized in terms of types of rings:

    rings = np.array([[[[0,0,0], [0, 1, 0]], [[1, -1, 0], [2, 0, 0]], [[0, -1, 0], [1, 2, 0]]],
                      [[[0,0,0], [0, 2, 0]], [[-1, 1, 0], [1, 0, 0]], [[-1, 0, 0], [2, 1, 0]]],
                      [[[0,0,0], [1, 2, 0]], [[0, 1, 0], [0, 1, 0]], [[1, 0, 0], [2, 0, 0]]],

                      [[[0,0,0], [1, 2, 0]], [[0, 1, -1], [3, 1, 0]], [[1, 0, -1], [2, 3, 0]]],
                      [[[0,0,0], [1, 3, 0]], [[0, -1, 1], [2, 1, 0]], [[1, -1, 0], [3, 2, 0]]],
                      [[[0,0,0], [2, 3, 0]], [[-1, 0, 1], [1, 2, 0]], [[-1, 1, 0], [3, 1, 0]]],

                      [[[0,0,0], [0, 2, 0]], [[0, 1, -1], [3, 0, 0]], [[0, 0, -1], [2, 3, 0]]],
                      [[[0,0,0], [0, 3, 0]], [[0, -1, 1], [2, 0, 0]], [[0, -1, 0], [3, 2, 0]]],
                      [[[0,0,0], [2, 3, 0]], [[0, 0, 1], [0, 2, 0]], [[0, 1, 0], [3, 0, 0]]],

                      [[[0,0,0], [0, 1, 0]], [[1, 0, -1], [3, 0, 0]], [[0, 0, -1], [1, 3, 0]]],
                      [[[0,0,0], [0, 3, 0]], [[-1, 0, 1], [1, 0, 0]], [[-1, 0, 0], [3, 1, 0]]],
                      [[[0,0,0], [1, 3, 0]], [[0, 0, 1], [0, 1, 0]], [[1, 0, 0], [3, 0, 0]]]])

    fluxs = np.zeros((4,3))

    FluxString = ["012", "123", "230", "301"]

    for i in range(12):
        a = i // 3
        b = i % 3
        # if i % 3 == 0:
            # print("Begins flux ring of type " + FluxString[a])
        for j in range(3):
            temp = getIndex(indexRule(site + rings[i,j,0], unitCell), unitCell)
            # if j == 2:
            #     print("A"+str(temp)+str(rings[i,j,1,0])+" + A"+str(temp)+str(rings[i,j,1,1]), end=";")
            # else:
            #     print("A"+str(temp)+str(rings[i,j,1,0])+" + A"+str(temp)+str(rings[i,j,1,1])+" + ", end="")
            for k in range(2):
                fluxs[a, b] = fluxs[a, b] + A_pi[temp, rings[i,j,1,k]]
        # print("")

    # fluxs = np.mod(fluxs, 2*np.pi)
    # fluxs = np.where(fluxs > np.pi, fluxs-2*np.pi, fluxs)
    return fluxs

def totalFlux(unitCell, A_pi):
    A = np.zeros((4,4,3))
    for i in range(4):
        A[i] = calFlux(unitCell[i], A_pi, unitCell)
        # print('--------------------------------')
    return np.mod(A,2*np.pi)

def constructA_pi_012(Astart):
    A00, A01, A02, A03, A10, A20, A21 = Astart
    A11 = A01
    A12 = A00 + A02 - A10
    A13 = A03
    A22 = A02
    A23 = A00 + A03 + A11 - A20 - A21
    A30 = -A00 + A10 + A20
    A31 = A01 - A11 + A21
    A32 = A00 + A02 - A10
    A33 = A00 + A03 + A11 - A20 - A21

    return np.array([[A00, A01, A02, A03],
                     [A10, A11, A12, A13],
                     [A20, A21, A22, A23],
                     [A30, A31, A32, A33]])

def constructA_pi(flux, A00 = 0, A01 = 0, A02 = 0, A10 = 0):
    A, B, C, D = flux

    A03 = -A00 - A01 - A02 + (A + B + C + D) / 6
    A11 = A - A00 - A01 - 2 * A02 - A10
    A12 = A02
    A13 = A00 + A01 + A02 + (-5 * A + B + C + D) / 6
    A20 = -A10 + (A - 2 * B + C + D) / 3
    A21 = A00 + A01 + 2 * A02 + A10 + (-2 * A + B - 2 * C + D) / 3
    A22 = -A02 + (A + B + C - 2 * D) / 3
    A23 = -A00 - A01 - A02 + (A + B + C + D) / 6
    A30 = -A00 + (A - 2 * B + C + D) / 3
    A31 = -A01 + (A + B - 2 * C + D) / 3
    A32 = -A02 + (A + B + C - 2 * D) / 3
    A33 = A00 + A01 + A02 + (-5 * A + B + C + D) / 6
    M = np.array([[A00, A01, A02, A03],
                 [A10, A11, A12, A13],
                 [A20, A21, A22, A23],
                 [A30, A31, A32, A33]])
    return M

def constructA_pi_0(flux, A01 = 0):
    A, B, C, D = flux

    A00 = (A - 2 * B + C + D) / 6
    A02 = -A01 + (2 * A + 2 * B - C - D) / 6
    A03 = (-2 * A + B + C + D) / 6
    A10 = (A - 2 * B + C + D) / 6
    A11 = A01
    A12 = -A01 + (2 * A + 2 * B - C - D) / 6
    A13 = (-2 * A + B + C + D) / 6
    A20 = (A - 2 * B + C + D) / 6
    A21 = -A01 + (A + B - 2 * C + D) / 3
    A22 = A01 + (C - D) / 2
    A23 = (-2 * A + B + C + D) / 6
    A30 = (A - 2 * B + C + D) / 6
    A31 = -A01 + (A + B - 2 * C + D) / 3
    A32 = A01 + (C - D) / 2
    A33 = (-2 * A + B + C + D) / 6

    M = np.array([[A00, A01, A02, A03],
                 [A10, A11, A12, A13],
                 [A20, A21, A22, A23],
                 [A30, A31, A32, A33]])
    return M


def Ainit(Fluxs, A00=0, A01=0, A02=0, A10=0):
    Fluxs = np.mod(Fluxs, 2*np.pi)
    Fluxs = np.where(Fluxs>np.pi, Fluxs-2*np.pi, Fluxs)
    A, B, C, D = Fluxs
    A03 = -A00 - A01 - A02 + (A+B+C+D)/6
    A11 = A - A00 - A01 - 2*A02 - A10
    A20 = -A10 + (A-2*B+C+D)/3
    A21 = A00 + A01 + 2*A02 + A10 + (-2*A+B-2*C+D)/3
    return np.array([A00, A01, A02, A03, A10, A11, A20, A21])


def constructA_pi_110(Fluxs):
    n1, n2, n4 = 1, 1, 0

    A00, A01, A03,F = A_init110(Fluxs, n1, n2)
    A02 = A01 + n4*np.pi + F

    A10 = A00
    A20 = A00
    A30 = A00
    A11 = A01 + n1*np.pi
    A21 = A01 + n2*np.pi
    A31 = A01 + (n1+n2) * np.pi
    A12 = A02
    A22 = A02 + n2*np.pi
    A32 = A02 + n2*np.pi
    A13 = A03
    A23 = A03
    A33 = A03
    M =  np.array([[A00, A01, A02, A03],
                     [A10, A11, A12, A13],
                     [A20, A21, A22, A23],
                     [A30, A31, A32, A33]])
    return np.mod(M, 2*np.pi)

def A_init110(Fluxs, n1, n2):
    A, B, C, D = Fluxs-np.pi*np.array([n1,n1,n2,n2])
    A00 = (A - 2 * B + C + D) / 6
    A01 = (A + B - 2 * C + D) / 6
    A03 = (-2 * A + B + C + D) / 6
    F = (C - D) / 2
    return np.array([A00,A01,A03,F])

def constructA_pi_001(Fluxs):

    n1, n2, n4, n5 = 1, 1, 0,0
    A00, A01 = A_init001(Fluxs, n1, n2)

    A03 = A00 + (n4+n5)*np.pi
    A02 = A01 + n5*np.pi

    A10 = A00
    A20 = A00
    A30 = A00
    A11 = A01 + n1*np.pi
    A21 = A01 + n1*np.pi
    A31 = A01
    A12 = A02
    A22 = A02 + n2*np.pi
    A32 = A02 + n2*np.pi
    A13 = A03
    A23 = A03
    A33 = A03

    return np.array([[A00, A01, A02, A03],
                     [A10, A11, A12, A13],
                     [A20, A21, A22, A23],
                     [A30, A31, A32, A33]])

def A_init001(Fluxs, n1, n2):
    ## A=B, C=D
    A, B, C, D = Fluxs - np.array([n1,n2,n2,n1])*np.pi

    A00 = (-A + 2*C)/6
    A01 = (2*A - C)/6
    return np.array([A00,A01])

def constructA_pi_111(Fluxs):
    n1, n5 = 1, 0

    A00, A01, F = A_init111(Fluxs)
    A02 = A01 + n5*np.pi
    A03 = A01 + n5*np.pi + F

    A10 = A00
    A20 = A00
    A30 = A00
    A13 = A03
    A23 = A03
    A33 = A03

    A11 = A01 + n1*np.pi
    A21 = A01 + n1*np.pi
    A31 = A01

    A12 = A02
    A22 = A02 + n1*np.pi
    A32 = A02 + n1*np.pi

    return np.array([[A00, A01, A02, A03],
                     [A10, A11, A12, A13],
                     [A20, A21, A22, A23],
                     [A30, A31, A32, A33]])

def A_init111(Fluxs):
    ## C=D
    A, B, C, D = Fluxs - np.ones(4)*np.pi

    A00 = (A - 2 * B + 2 * D) / 6
    A01 = (A + B - D) / 6
    F = (-A + D) / 2
    return np.array([A00,A01,F])
