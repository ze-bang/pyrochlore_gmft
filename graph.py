import numpy as np
import matplotlib.pyplot as plt

# JP, zgaps, zlambdas, zGS, pgaps, plambdas, pGS = np.loadtxt("test2.txt", delimiter=' ')
#
# plt.plot(JP, 0.5-zlambdas, JP, 0.5-plambdas)
# plt.legend(["Zero Flux", "Pi Flux"])
# plt.ylabel(r'$GS$')
# plt.xlabel(r'$J_{\pm}/J_{zz}$')
# plt.show()




def graphPhase(filename):
    phases = np.loadtxt(filename, delimiter=' ').T

    tempc = np.copy(phases[:phases.shape[0]-1, :])
    tempc = np.flip(tempc, axis=0)

    for i in range(tempc.shape[0]):
        for j in range(tempc.shape[1]):
            if (tempc[i][j] == 3) or (tempc[i][j] == -3):
                pass
            else:
                tempc[i][j] = -tempc[i][j]

    # print(phases)
    # print(tempc)

    bigphase = np.concatenate((phases, tempc), axis=0)


    JP = np.linspace(-0.5, 0.1, phases.shape[1])
    kappa = np.linspace(-1, 0, phases.shape[0])
    tempk = np.copy(kappa[:phases.shape[0]-1] )
    tempk = -np.flip(tempk)
    bigkappa = np.concatenate((kappa, tempk), axis=0)

    X,Y=np.meshgrid(JP, bigkappa)

    # plt.imshow(bigphase, cmap='gray', vmin=-3, vmax=3, interpolation='bilinear', extent=[-0.1, 0.1, -1, 1], aspect='auto')

    plt.contourf(X, Y, bigphase)
    plt.xlabel(r'$J_\pm/J_{zz}$')
    plt.ylabel(r'$(\kappa-1)/(\kappa+1)$')
    plt.show()


def graphMagPhase(filename):
    phases = np.loadtxt(filename, delimiter=' ').T


    JP = np.linspace(0, 0.25, phases.shape[1])
    h = np.linspace(0, 3, phases.shape[0])

    X,Y=np.meshgrid(JP, h)

    # plt.imshow(bigphase, cmap='gray', vmin=-3, vmax=3, interpolation='bilinear', extent=[-0.1, 0.1, -1, 1], aspect='auto')

    plt.contourf(X, Y, phases)
    plt.xlabel(r'$J_\pm/J_{y}$')
    plt.ylabel(r'$h/J_{y}$')
    plt.show()


# graphPhase("Files/phase_diagram.txt")
graphMagPhase("phase_mag_111.txt")