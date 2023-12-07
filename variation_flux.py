import matplotlib.pyplot as plt
import warnings
from misc_helper import *
from flux_stuff import *
import pyrochlore_general as pygen

# n = h111
ringv = np.array([[1 ,1 ,-1] ,[-1 ,-1 ,-1] ,[-1 ,1 ,1] ,[1 ,-1 ,1]] ) /np.sqrt(3)
# transf = np.around(contract('ij,j->i', ringv, n), decimals=15)
# transcount = 0
# transformation = []
# while True:
#     a = np.where(transf==transf[transcount])[0]
#     transformation = transformation + [a]
#     transcount = transcount + 1
#     if size_non_h(transformation) == 4:
#         break
# transformation = non_h_unique(transformation)

def gradient_flux(Jxx, Jyy, Jzz, h, n, kappa, BZres, flux):
    tol = 1e-6
    p0 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux)
    p1 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux+tol*np.array([1,0,0,0]))
    p2 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux+tol*np.array([0,1,0,0]))
    p3 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux+tol*np.array([0,0,1,0]))
    p4 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux+tol*np.array([0,0,0,1]))
    p0.solvemeanfield()
    p1.solvemeanfield()
    p2.solvemeanfield()
    p3.solvemeanfield()
    p4.solvemeanfield()
    p0M = p0.MFE()

    return np.array([p1.MFE()-p0M, p2.MFE()-p0M, p3.MFE()-p0M, p4.MFE()-p0M])/tol


def findflux(Jxx, Jyy, Jzz, h, n, kappa, BZres):
    step = 1
    flux = np.around(contract('ij,j->i', ringv, n), decimals=15)
    init = True
    while True:
        if not init:
            gradnow = gradient_flux(Jxx, Jyy, Jzz, h, n, kappa, BZres, flux)
            gradlen = gradnow - gradlast
            try:
                step = abs(np.dot(flux - fluxlast, gradlen)) / np.linalg.norm(gradlen) ** 2
            except:
                step = 0
            fluxlast = np.copy(flux)
            gradlast = np.copy(gradnow)
            flux = flux - step * gradlast
            init = False
        else:
            fluxlast = np.copy(flux)
            gradlast = gradient_flux(Jxx, Jyy, Jzz, h, n, kappa, BZres, flux)
            flux = flux - step * gradlast
            init = False
        if (abs(flux - fluxlast) < 1e-7).all():
            break
    return flux


