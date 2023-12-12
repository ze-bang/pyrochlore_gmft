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

def gradient_flux(Jxx, Jyy, Jzz, h, n, kappa, BZres, flux, p0M):
    tol = 1e-6
    p1 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux+tol*np.array([1,0,0,0]))
    p2 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux+tol*np.array([0,1,0,0]))
    p3 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux+tol*np.array([0,0,1,0]))
    p4 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, graphres=graphres, BZres=BZres, h=h, n=n, flux=flux+tol*np.array([0,0,0,1]))
    p1.solvemeanfield()
    p2.solvemeanfield()
    p3.solvemeanfield()
    p4.solvemeanfield()

    return np.array([p1.MFE()-p0M, p2.MFE()-p0M, p3.MFE()-p0M, p4.MFE()-p0M])/tol

def fluxMFE(flux, Jxx, Jyy, Jzz, h, n, kappa, BZres):
    p0 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, BZres=BZres, h=h, n=n,
                            flux=flux)
    p0.solvemeanfield()
    return p0.MFE()

def findflux(Jxx, Jyy, Jzz, h, n, kappa, BZres, fluxstart):
    step = 1
    flux = fluxstart
    init = True
    while True:
        if not init:
            temp = fluxMFE(flux, Jxx, Jyy, Jzz, h, n, kappa, BZres)
            gradnow = gradient_flux(Jxx, Jyy, Jzz, h, n, kappa, BZres, flux, temp)
            gradlen = gradnow - gradlast
            try:
                step = abs(np.dot(flux - fluxlast, gradlen)) / np.linalg.norm(gradlen) ** 2
            except:
                step = 0
            fluxlast = np.copy(flux)
            gradlast = np.copy(gradnow)
            pmlast = pm
            pm = temp
            flux = flux - step * gradlast
            flux = np.mod(flux, 2 * np.pi)
            # flux = np.where(flux > np.pi, flux - 2 * np.pi, flux)
            # print(flux, pm, pmlast)
            if (abs(pm - pmlast) < 1e-10).all():
                break
        else:
            fluxlast = np.copy(flux)
            pm = fluxMFE(flux, Jxx, Jyy, Jzz, h, n, kappa, BZres)
            gradlast = gradient_flux(Jxx, Jyy, Jzz, h, n, kappa, BZres, flux, pm)
            flux = flux - step * gradlast
            flux = np.mod(flux, 2 * np.pi)
            # flux = np.where(flux > np.pi, flux - 2 * np.pi, flux)
            init = False
    return flux, pm

def flux_converge(h, hat):
    n = 4
    fluxs = np.zeros((n,4))
    mfes = np.zeros(n)
    for i in range(n):
        fluxs[i], mfes[i] = findflux(0, 0, 1, h, hat, 2, 26, np.random.rand(4))
    print(fluxs)
    a = np.argmin(mfes)
    return fluxs[a]

def flux_converge_scipy(Jxx, Jyy, Jzz, h, hat, kappa, BZres, n):
    fluxs = np.zeros((n,4))
    mfes = np.zeros(n)
    for i in range(n):
        res = minimize(fluxMFE, np.random.rand(4), args=(Jxx, Jyy, Jzz, h, hat, kappa, BZres), method='Nelder-Mead', bounds=((0,2*np.pi), (0,2*np.pi),(0,2*np.pi),(0,2*np.pi)))
        fluxs[i] = np.array(res.x)
        mfes[i] = res.fun
    print(fluxs)
    a = np.argmin(mfes)
    return fluxs[a]

flux_converge_scipy(0, 0, 1, 0.3, h111, 2, 26)
flux_converge_scipy(0, 0, 1, 0.3, h110, 2, 26)
flux_converge_scipy(0, 0, 1, 0.3, h001, 2, 26)



