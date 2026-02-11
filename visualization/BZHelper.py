from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numba as nb

# function for z axis
BasisBZA = np.array([2*np.pi*np.array([-1,1,1]),2*np.pi*np.array([1,-1,1]),2*np.pi*np.array([1,1,-1])])

@nb.njit
def genBZ(d):
    # d = d*1j
    temp = np.zeros((d*d*d, 3))
    s = np.linspace(0,1,d)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                temp[i*d*d+j*d+k] = s[i]*BasisBZA[0] + s[j]*BasisBZA[1] + s[k]*BasisBZA[2]
    return temp

def ifFBZ(k):
    b1, b2, b3 = abs(k)
    if np.any(abs(k) > 2*np.pi):
        return False
    if abs(b1+b2+b3)<3*np.pi and abs(b1-b2+b3)<3*np.pi and abs(-b1+b2+b3)<3*np.pi and abs(b1+b2-b3)<3*np.pi:
        return True
    else:
        return False
#

def genBZA(d):
    d = d*1j
    b = np.mgrid[-2*np.pi:2*np.pi:d, -2*np.pi:2*np.pi:d, -2*np.pi:2*np.pi:d].reshape(3,-1).T
    BZ = []
    for x in b:
        if ifFBZ(x)==True:
            BZ += [x]
    return np.array(BZ)


X, Y, Z = genBZ(20).T
A, B, C = genBZA(20).T

print(A.shape)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z, color='green')
ax.scatter(A, B, C, color='blue')


plt.show()