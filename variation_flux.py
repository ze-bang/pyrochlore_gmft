import matplotlib.pyplot as plt
from misc_helper import *
from flux_stuff import *
import pyrochlore_general as pygen
from phase_diagram import graphMagPhase

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

def fluxMFE_110(flux, Jxx, Jyy, Jzz, h, n, kappa, BZres):
    p0 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, BZres=BZres, h=h, n=n,
                            flux=np.array([flux[0], flux[0], flux[1], flux[1]]))
    if p0.validgauge:
        p0.solvemeanfield()
        return p0.MFE()
    else:
        return np.NaN

def fluxMFE_111(flux, Jxx, Jyy, Jzz, h, n, kappa, BZres):
    p0 = pygen.piFluxSolver(Jxx, Jyy, Jzz, kappa=kappa, BZres=BZres, h=h, n=n,
                            flux=np.array([flux[0], flux[1], flux[0], flux[0]]))
    if p0.validgauge:
        p0.solvemeanfield()
        return p0.MFE()
    else:
        return np.NaN

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

def flux_converge(h, hat, n):
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
    a = np.argmin(mfes)
    return fluxs[a] 

def flux_converge_scipy_110(Jxx, Jyy, Jzz, h, hat, kappa, BZres, n):
    fluxs = np.zeros((n,2))
    mfes = np.zeros(n)
    for i in range(n):
        res = minimize(fluxMFE_110, np.random.rand(2), args=(Jxx, Jyy, Jzz, h, hat, kappa, BZres), method='Nelder-Mead', bounds=((0,np.pi), (-np.pi,np.pi)))
        fluxs[i] = np.array(res.x)
        mfes[i] = res.fun
    a = np.argmin(mfes)
    return fluxs[a]

def flux_converge_scipy_111(Jxx, Jyy, Jzz, h, hat, kappa, BZres, n):
    fluxs = np.zeros((n,2))
    mfes = np.zeros(n)
    for i in range(n):
        res = minimize(fluxMFE_111, np.random.rand(2), args=(Jxx, Jyy, Jzz, h, hat, kappa, BZres), method='Nelder-Mead', bounds=((0,np.pi), (-np.pi,np.pi)))
        fluxs[i] = np.array(res.x)
        mfes[i] = res.fun
    a = np.argmin(mfes)
    return fluxs[a]

def flux_converge_line(Jmin, Jmax, nJ, h, hat, kappa, BZres, n, filename):
    JP = np.linspace(Jmin, Jmax, nJ)
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nb = nJ/size
    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK
    currJP = JP[leftK: rightK]

    sendtemp = np.zeros((currsizeK, 2))
    rectemp = None

    if rank == 0:
        rectemp = np.zeros((nJ, 2))

    for i in range(currsizeK):
        if (hat == h111).all():
            sendtemp[i] = flux_converge_scipy_111(-2*currJP[i], -2*currJP[i], 1, h, hat, kappa, BZres, n)
        elif (hat == h110).all():
            sendtemp[i] = flux_converge_scipy_110(-2*currJP[i], -2*currJP[i], 1, h, hat, kappa, BZres, n)

    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)

    if rank == 0:
        toSave = np.zerso((n, 3))
        toSave[:,0] = JP
        toSave[:,1:-1] = rectemp
        np.savetxt('Files/' + filename + '.txt', toSave)

def plot_MFE_flux_110(Jxx, Jyy, Jzz, h, hat, kappa, BZres, n, filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    fluxplane = np.mgrid[-np.pi:np.pi:1j*n, -np.pi:np.pi:1j*n].reshape(2,-1).T

    le = n**2
    nb = le/size
    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK
    currFlux = fluxplane[leftK:rightK, :]
    sendtemp = np.zeros(currsizeK, dtype=np.float64)

    rectemp = None
    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)

    for i in range(currsizeK):
        sendtemp[i] = fluxMFE_110(currFlux[i], Jxx, Jyy, Jzz, h, hat, kappa, BZres)

    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    
    if rank == 0:
        rectemp = rectemp.reshape((n, n))
        np.savetxt('Files/' + filename+'.txt', rectemp)
        FD = np.linspace(-np.pi,np.pi,n)
        X,Y = np.meshgrid(FD, FD)

        plt.pcolormesh(X, Y, rectemp.T)
        plt.colorbar()
        plt.xlabel(r'$F_\alpha$')
        plt.ylabel(r'$F_\beta$')
        plt.savefig('Files/' + filename +'.png')
        plt.clf()


def plot_MFE_flux_111(Jxx, Jyy, Jzz, h, hat, kappa, BZres, n, filename):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    fluxplane = np.mgrid[-np.pi:np.pi:1j*n, -np.pi:np.pi:1j*n].reshape(2,-1).T

    le = n**2
    nb = le/size
    leftK = int(rank*nb)
    rightK = int((rank+1)*nb)
    currsizeK = rightK-leftK
    currFlux = fluxplane[leftK:rightK, :]
    sendtemp = np.zeros(currsizeK, dtype=np.float64)

    rectemp = None
    if rank == 0:
        rectemp = np.zeros(le, dtype=np.float64)

    for i in range(currsizeK):
        sendtemp[i] = fluxMFE_111(currFlux[i], Jxx, Jyy, Jzz, h, hat, kappa, BZres)

    sendcounts = np.array(comm.gather(sendtemp.shape[0], 0))
    comm.Gatherv(sendbuf=sendtemp, recvbuf=(rectemp, sendcounts), root=0)
    
    if rank == 0:
        rectemp = rectemp.reshape((n, n))
        np.savetxt('Files/' + filename+'.txt', rectemp)
        FD = np.linspace(-np.pi,np.pi,n)
        X,Y = np.meshgrid(FD, FD)

        plt.pcolormesh(X, Y, rectemp.T)
        plt.colorbar()
        plt.xlabel(r'$F_\alpha$')
        plt.ylabel(r'$F_\beta$')
        plt.savefig('Files/' + filename +'.png')
        plt.clf()



def plot_MFE_flux(Jxx, Jyy, Jzz, h, hat, kappa, BZres, n, filename):
    if (hat == h111).all():
        return plot_MFE_flux_111(Jxx, Jyy, Jzz, h, hat, kappa, BZres, n, filename)
    elif (hat == h110).all():
        return plot_MFE_flux_110(Jxx, Jyy, Jzz, h, hat, kappa, BZres, n, filename)






