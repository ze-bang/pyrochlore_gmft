import numpy as np
from opt_einsum import contract
#Stevens Operator:

#X = J(J+1)

#20 = 3Jz^2 - X
#22 = (J+^2 + J-^2)/2
#2-2 = (J+^2 - J-^2)/2i

#40 = 35Jz^4 - (30X-25)Jz^2 + 3X^2 - 6X
#42 = 1/4*((J+^2+J-^2)(7Jz^2-X-5)+(7Jz^2-X-5)(J+^2+J-^2))
#4-2 = 1/4i*((J+^2-J-^2)(7Jz^2-X-5)+(7Jz^2-X-5)(J+^2-J-^2))
#44 = 1/2(J+^4+J-^4)
#4-4 = 1/2i(J+^4-J-^4)

#60 = 231Jz^6 - (315X-735)Jz^4 + (105X^2-525X+294)Jz^2 - 5X^3 + 40X^2 - 60X
#62 = 1/4*((J+^2+J-^2)(33Jz^4 - (18X+123)Jz^2 + X^2 + 10X +102) + (33Jz^4 - (18X+123)Jz^2 + X^2 + 10X +102)(J+^2+J-^2))
#6-2 = 1/4i*((J+^2-J-^2)(33Jz^4 - (18X+123)Jz^2 + X^2 + 10X +102) + (33Jz^4 - (18X+123)Jz^2 + X^2 + 10X +102)(J+^2-J-^2))
#64 = 1/4*((J+^4+J-^4)(11Jz^2-X-38)+(11Jz^2-X-38)(J+^4+J-^4))
#6-4 = 1/4i*((J+^4-J-^4)(11Jz^2-X-38)+(11Jz^2-X-38)(J+^4-J-^4))
#66 = 1/2*(J+^6+J-^6)
#6-6 = 1/2i*(J+^6-J-^6)


#Now we want to get the matrix element for the states |J, Jz> where J=6. Which means a 13x13 states.
#Matrix element O_{ij} = <Jz_i|O|Jz_j>. Jz|Jz_j> = Jz_j|Jz_j>

def op_add(O1, O2, state):
    return O1(state) + O2(state)

def op_mult(O1, O2, state):
    return O1(O2(state))

def op_pow(O1, n, state):
    temp = np.copy(state)
    for i in range(n):
        temp = O1(temp)
    return temp

J = 6
X = J*(J+1)
Jz_index = np.arange(-6,7,dtype=int)

def Jplus(state):
    temp_J = np.zeros_like(Jz_index, dtype=np.complex128)

    Jplus_indx = np.zeros(13)
    Jplus_indx[1:] = np.arange(-6,6,dtype=int)

    Jplux_index = np.sqrt(X - Jplus_indx*(Jplus_indx+1))

    temp_J[1:] = state[0:-1]
    return Jplux_index*temp_J
def Jminus(state):
    temp_J = np.zeros_like(Jz_index, dtype=np.complex128)

    Jplus_indx = np.zeros(13)
    Jplus_indx[:-1] = np.arange(-5,7,dtype=int)

    Jplux_index = np.sqrt(X - Jplus_indx*(Jplus_indx-1))

    temp_J[:-1] = state[1:]
    return Jplux_index*temp_J
def Jz(state):
    return Jz_index*state
def XJ(state):
    return X*state
def Iden(state, const=1):
    return const*state



#20 = 3Jz^2 - X
def O20(state):
    return 3*op_pow(Jz, 2, state) - XJ(state)

#22 = (J+^2 + J-^2)/2
def O22(state):
    return (op_pow(Jplus, 2, state) + op_pow(Jminus, 2, state))/2
#2-2 = (J+^2 - J-^2)/2i
def O2n2(state):
    return (op_pow(Jplus, 2, state) - op_pow(Jminus, 2, state))/(2j)


#40 = 35Jz^4 - (30X-25)Jz^2 + 3X^2 - 6X
def O40(state):
    return 35*op_pow(Jz, 4, state) - (30*X-25)*op_pow(Jz, 2, state) + (3*X**2-6*X)*Iden(state)

#42 = 1/4*((J+^2+J-^2)(7Jz^2-X-5)+(7Jz^2-X-5)(J+^2+J-^2))

def O42(state):
    S1 = 7*op_pow(Jz, 2, state) - XJ(state) - 5*Iden(state)
    S2 = op_pow(Jplus, 2, state) + op_pow(Jminus, 2, state)
    return (op_pow(Jplus, 2, S1) + op_pow(Jminus, 2, S1) + 7*op_pow(Jz, 2, S2) - XJ(S2) - 5*Iden(S2))/4

#4-2 = 1/4i*((J+^2-J-^2)(7Jz^2-X-5)+(7Jz^2-X-5)(J+^2-J-^2))
def O4n2(state):
    S1 = 7*op_pow(Jz, 2, state) - XJ(state) - 5*Iden(state)
    S2 = op_pow(Jplus, 2, state) - op_pow(Jminus, 2, state)
    return (op_pow(Jplus, 2, S1) - op_pow(Jminus, 2, S1) + 7*op_pow(Jz, 2, S2) - XJ(S2) - 5*Iden(S2))/(4j)

#44 = 1/2(J+^4+J-^4)

def O44(state):
    return (op_pow(Jplus, 4, state) + op_pow(Jminus, 4, state))/2

#4-4 = 1/2i(J+^4-J-^4)

def O4n4(state):
    return (op_pow(Jplus, 4, state) - op_pow(Jminus, 4, state))/(2j)


#60 = 231Jz^6 - (315X-735)Jz^4 + (105X^2-525X+294)Jz^2 - 5X^3 + 40X^2 - 60X

def O60(state):
    return 231*op_pow(Jz, 6, state) - (315*X-735)*op_pow(Jz, 4, state) \
            + (105*X**2 - 525*X + 294)*op_pow(Jz, 2, state) + (-5*X**3+40*X**2-60*X)*Iden(state)

#62 = 1/4*((J+^2+J-^2)(33Jz^4 - (18X+123)Jz^2 + X^2 + 10X +102) + (33Jz^4 - (18X+123)Jz^2 + X^2 + 10X +102)(J+^2+J-^2))

def O62(state):
    S1 = 33*op_pow(Jz, 4, state) - (18*X+123)*op_pow(Jz, 2, state) + (X**2 + 10*X + 102)*Iden(state)
    S2 = op_pow(Jplus, 2, state) + op_pow(Jminus, 2, state)
    return (op_pow(Jplus, 2, S1) + op_pow(Jminus, 2, S1) + \
        33 * op_pow(Jz, 4, S2) - (18 * X + 123) * op_pow(Jz, 2, S2) + (X ** 2 + 10 * X + 102) * Iden(S2))/4
#6-2 = 1/4i*((J+^2-J-^2)(33Jz^4 - (18X+123)Jz^2 + X^2 + 10X +102) + (33Jz^4 - (18X+123)Jz^2 + X^2 + 10X +102)(J+^2-J-^2))

def O6n2(state):
    S1 = 33*op_pow(Jz, 4, state) - (18*X+123)*op_pow(Jz, 2, state) + (X**2 + 10*X + 102)*Iden(state)
    S2 = op_pow(Jplus, 2, state) - op_pow(Jminus, 2, state)
    return (op_pow(Jplus, 2, S1) - op_pow(Jminus, 2, S1) + \
        33 * op_pow(Jz, 4, S2) - (18 * X + 123) * op_pow(Jz, 2, S2) + (X ** 2 + 10 * X + 102) * Iden(S2))/(4j)

#64 = 1/4*((J+^4+J-^4)(11Jz^2-X-38)+(11Jz^2-X-38)(J+^4+J-^4))

def O64(state):
    S1 = 11*op_pow(Jz, 2, state) - (X+38)*Iden(state)
    S2 = op_pow(Jplus, 4, state) + op_pow(Jminus, 4, state)
    return (op_pow(Jplus, 4, S1) + op_pow(Jminus, 4, S1) + 11*op_pow(Jz, 2, S2) - (X+38)*Iden(S2))/4

#6-4 = 1/4i*((J+^4-J-^4)(11Jz^2-X-38)+(11Jz^2-X-38)(J+^4-J-^4))
def O6n4(state):
    S1 = 11*op_pow(Jz, 2, state) - (X+38)*Iden(state)
    S2 = op_pow(Jplus, 4, state) - op_pow(Jminus, 4, state)
    return (op_pow(Jplus, 4, S1) - op_pow(Jminus, 4, S1) + 11*op_pow(Jz, 2, S2) - (X+38)*Iden(S2))/(4j)

#66 = 1/2*(J+^6+J-^6)

def O66(state):
    return (op_pow(Jplus, 6, state) + op_pow(Jminus, 6, state))/2

#6-6 = 1/2i*(J+^6-J-^6)
def O6n6(state):
    return (op_pow(Jplus, 6, state) - op_pow(Jminus, 6, state))/(2j)


def construct_matrix(O):
    MtoReturn = np.zeros((13,13),dtype=np.complex128)
    for i in range(13):
        bra = np.zeros(13)
        bra[i] = 1
        for j in range(13):
            ket = np.zeros(13)
            ket[j] = 1
            Oj = O(ket)
            MtoReturn[i,j] = np.dot(bra, Oj)
    return MtoReturn


B20, B22, B2n2, B40, B42, B4n2, B44, B4n4, B60, B62, B6n2, B64, B6n4, B66, B6n6 = -5.29e-1, -1.35e-1, 12.79e-1, -0.13e-3, -1.7e-3, 3.29e-3, -1.22e-3, -9.57e-3, 0.2e-5, -1.1e-5, -0.9e-5, 6.1e-5, 0.3e-5, -0.9e-5, 0

# B20, B22, B2n2, B40, B42, B4n2, B44, B4n4, B60, B62, B6n2, B64, B6n4, B66, B6n6 = -5.29e-1, -1.35e-1, 0, -0.13e-3, -1.7e-3, 0, -1.22e-3, 0, 0.2e-5, -1.1e-5, 0, 6.1e-5, 0, -0.9e-5, 0


H = B20*construct_matrix(O20) + B22*construct_matrix(O22) + B2n2*construct_matrix(O2n2) \
    + B40*construct_matrix(O40) + B42*construct_matrix(O42) + B4n2*construct_matrix(O4n2) \
    + B44 * construct_matrix(O44) + B4n4 * construct_matrix(O4n4) + B60 * construct_matrix(O60) \
    + B62 * construct_matrix(O62) + B6n2 * construct_matrix(O6n2) + B64 * construct_matrix(O64) \
    + B6n4 * construct_matrix(O6n4) + B66 * construct_matrix(O66) + B6n6 * construct_matrix(O6n6)


Z = H - np.transpose(H)

M = (H + np.transpose(np.conj(H))) / 2

E, V = np.linalg.eigh(H)
E_adj = E - np.min(E)
E0 = V[:, 0]
E1 = V[:, 1]
E2 = V[:, 2]
Vabs = np.abs(V)
Vangle = np.angle(V)
T = np.zeros((13,3), dtype=np.complex128)
for i in range(3):
    if i == 2:
        Angle = (Vangle[1,i]+Vangle[11,i])/2
        rot = np.exp(-1j*Angle)
    else:
        Angle = (Vangle[0, i] + Vangle[12, i]) / 2
        rot = np.exp(-1j * Angle)
    T[:,i] = rot*V[:,i]

E0_paper = np.array([-0.565 + 0.130j, 0, 0.029 + 0.322j,0, 0.202 - 0.069j, 0, -(0.135 + 0.103j), 0, -0.013 + 0.213j, 0, 0.318 - 0.058j, 0, -(0.024 - 0.579j)])
E1_paper = np.array([0.382 - 0.508j, 0, -(0.196 + 0.209j),0,-0.072 + 0.094j, 0, 0.018j, 0, 0.078 + 0.089j, 0, 0.182 - 0.221j, 0, - 0.414 - 0.482j])


E0_angle = np.angle(E0_paper)
Angle = (E0_angle[0] + E0_angle[12]) / 2
rot = np.exp(-1j * Angle)
E0_paper = rot*E0_paper

E1_angle = np.angle(E1_paper)
Angle = (E1_angle[0] + E1_angle[12]) / 2
rot = np.exp(-1j * Angle)
E1_paper = rot*E1_paper
# print(E0_paper)
# print(T[:,0])
# print(E0_paper)
# print(T[:,0])
# print(np.abs(E0_paper))
# print(T[:,0])
# print(V[:,0], V[:, 1], V[:, 2])
# print(E)
# print(T[:,0], T[:, 1], T[:, 2])
# print(contract('ia, a, i->i', H, T[:,2], 1/T[:,2]))
# print(contract('ia, a, i->i', H, E2, 1/E2))
# print(E0_paper + T[:,0])
# print(contract('ia, a, i->i', H, E0_paper, 1/E0_paper))
A = T[:,0]
B = T[:,1]
print(Jz(A))
print(B)
# print(Jz(T[:,0]))
# print(T[:,1])

print(np.abs(np.dot(B, op_add(Jplus, Jminus, A))/2)**2)
print(np.abs(np.dot(B, (Jplus(A)-Jminus(A)))/2j)**2)
print(np.abs(np.dot(B, Jz(A)))**2)


