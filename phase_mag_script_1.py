import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# import numpy as np
from phase_diagram import *
from misc_helper import *
from flux_stuff import *
from observables import *



# findPhaseMag111_ex(-0.5, 0.1, 100, 0, 1, 100, h111, 17, 2, "phase_111_kappa=2_ex")
# findPhaseMag111(-0.5, 0.1, 100, 0, 1, 100, h111, 30, 2, "phase_111_kappa=2")

# plotLinefromnetCDF(h111, "h111=0.1_largeN.pdf", h=0.1, diff=True)
# plotLinefromnetCDF(h111, "h111=0.2_largeN.pdf", h=0.2, diff=True
# plotLinefromnetCDF(h111, "h111=0.3_largeN.pdf", h=0.3, diff=True)
#
# plotLinefromnetCDF(h110, "h110=0.1_largeN.pdf", h=0.1, diff=True)
# plotLinefromnetCDF(h110, "h110=0.2_largeN.pdf", h=0.2, diff=True)
# plotLinefromnetCDF(h110, "h110=0.3_largeN.pdf", h=0.3, diff=True)
#
# plotLinefromnetCDF(h001, "h001=0.1_largeN.pdf", h=0.1, diff=True)
# plotLinefromnetCDF(h001, "h001=0.2_largeN.pdf", h=0.2, diff=True)
# plotLinefromnetCDF(h001, "h001=0.3_largeN.pdf", h=0.3, diff=True)

Jpm = 0.03
# py0s = pycon.piFluxSolver(-2*Jpm, -2*Jpm, 1, BZres=25, h=0.3, n=h111, flux=np.ones(4)*np.pi)
# py0s.solvemeanfield()
# print(SSSF_core(np.array([0.5,1,0.5]), hb110, py0s))
# print(SSSF_core(np.array([1,1,1]), hb110, py0s))
#
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.ones(4)*np.pi,30, "test")
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi,30, "test", "hhl")

#
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.zeros(4),30, "Files/SSSF/Jpm=0.02_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.02_pi")

# findXYZPhase(-1,1,-1,1,100,30,2,'XYZ_phase_diagram')
# DSSF(3, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.zeros(4), 30, "test")

# Jpm = 0.02
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h110=0")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h110=0.1")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h110=0.2")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h110=0.3")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.4, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h110=0.4")

# DSSF_pedantic(200,-2*Jpm, -2*Jpm, 1, 0.35, h111, np.zeros(4), 30, "Files/DSSF/Jpm=0.03_0/h111/h=0.35")
# kk = np.concatenate((GammaX, XGammap))
# Jpm = 0.045
# DSSF_pedantic_custom_Ks(200, kk, -2*Jpm, -2*Jpm, 1, 0.0, h111, np.zeros(4), 25, "DSSF/Jpm=0.045/")
# Jpm = -0.045
# DSSF_pedantic_custom_Ks(200, kk, -2*Jpm, -2*Jpm, 1, 0.0, h111, np.zeros(4), 25, "DSSF/Jpm=-0.045/")
# Jpm = -0.3
# DSSF_pedantic_custom_Ks(200, kk, -2*Jpm, -2*Jpm, 1, 0.0, h111, np.ones(4)*np.pi, 25, "DSSF/Jpm=-0.3/")
# DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h111, np.zeros(4), 30, "Files/DSSF/Jpm=0.03_0")
# DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=0.03_pi")
#
# # DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.14, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.14")
# # DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.16, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.16")
#
# Jpm=0.03
#
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 11, h111, np.zeros(4),30, "Files/SSSF/Jpm=0.03_0", "hh2k", 0, 3, 3)
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 11, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.03_pi", "hh2k", 0, 3, 3)
# Jxx = np.linspace(0,0.5,1 0)
# for i in range(10):
def QFI_at_K(X, n_sites):
    n = 2
    Szz = np.zeros(n)
    Sxx = np.zeros(n)
    T = np.logspace(0, 4, n)
    T = 1 / T
    # Jxx, Jyy, Jzz = 0.98412698412, 1.0, 0.1746031746
    Jxx, Jyy, Jzz = -0.09, -0.09, 1
    for i in range(n):
        print("Temp = ", T[i])
        tempzz, tempxx = SSSF_q_omega_beta_at_K(T[i], X, 500, Jxx, Jyy, Jzz, 0, h110, np.zeros(4), n_sites, 0)
        print("Szz = ", tempzz)
        print("Sxx = ", tempxx)
        Szz[i] = quantum_fisher_information_K(tempzz, T[i])[0]
        Sxx[i] = quantum_fisher_information_K(tempxx, T[i])[0]
    np.savetxt("QFI_T_Spm_local_X_L={}.dat".format(n_sites), np.column_stack((T, Szz)), header="T/|J_{zz}|^{-1} F_{QFI}[S^{pp}]")
    np.savetxt("QFI_T_Spp_local_X_L={}.dat".format(n_sites), np.column_stack((T, Sxx)), header="T/|J_{zz}|^{-1} F_{QFI}[S^{pm}]")

    # Szz = np.loadtxt("SSSF_0_flux_T=0_Szz_local_X.txt")
    # Sxx = np.loadtxt("SSSF_0_flux_T=0_Sxx_local_X.txt")
    plt.plot(T, Sxx, label="Spm")
    plt.plot(T, Szz, label="Spp")

    plt.legend([r"$F_{QFI}[S^{x}]$", r"$F_{QFI}[S^{y}]$"])
    plt.title(r"$F_{QFI}[S^{\alpha}]$  vs $T$ at $q=X$")
    plt.xlabel(r"$T/|J_{zz}|^{-1}$")
    plt.ylabel(r"$F_{QFI}$")
    plt.xscale("log")
    plt.savefig("SSSF_0_flux_T=0_X.pdf")
    plt.clf()


def QFI_at_K_T_zero(Jxx, Jyy, Jzz, X, flux, n_sites):

    tempzz, globalzz, tempxx, globalxx = SSSF_q_omega_at_K(X, Jxx, Jyy, Jzz, 0, h110, flux, n_sites, 0)
    Szz = 4*tempzz
    Szzglobal = 4*globalzz
    Sxx = 4*tempxx
    Sxxglobal = 4*globalxx
    print("Szz = ", tempzz)
    print("Szzglobal = ", globalzz)
    print("Sxx = ", tempxx)
    print("Sxxglobal = ", globalxx)
    return Szz, Szzglobal, Sxx, Sxxglobal

def QFI_scan(Xs, Xstring, n):
    len_Xs = len(Xs)
    Jpms = np.linspace(0.048, -0.5, n)
    Szz = np.zeros((n, len_Xs))
    Sxx = np.zeros((n, len_Xs))
    Szzglobal = np.zeros((n, len_Xs))
    Sxxglobal = np.zeros((n, len_Xs))
    for i in range(n):
        print("Jpm = ", Jpms[i])
        if Jpms[i] < 0:
            flux = np.ones(4) * np.pi
        else:
            flux = np.zeros(4)
        Jxx, Jyy, Jzz = -2*Jpms[i], -2*Jpms[i], 1
        Szz[i], Szzglobal[i], Sxx[i], Sxxglobal[i] = QFI_at_K_T_zero(Jxx, Jyy, Jzz, Xs, flux, 25)

    for j in range(len_Xs):
        plt.plot(Jpms, Szz[:, j], label=r"$F_{{QFI}}[S^{{\pm}}]$ at {}".format(Xstring[j]))
        plt.legend()
        # Create a safe filename by removing special characters
        safe_name = Xstring[j].replace("$", "").replace("\\", "").replace("'", "prime")
        plt.savefig("QFI_Jpm_Spinon_{}.pdf".format(safe_name))
        plt.clf()
        np.savetxt("QFI_Jpm_Spinon_Spm_{}.dat".format(safe_name), np.column_stack((Jpms, Szz[:, j])), header="Jpm/|Jzz| F_{QFI}[S^{pm}]")

QFI_scan(np.array([[0,0,0], [0.5, 0.5, 0], [1, 1, 0]]), np.array([r"$\Gamma$", r"X", r"$\Gamma^{\prime}$"]), 351)
# QFI_scan(np.array([[0.5, 0.5, 0.5]])*2*np.pi, "L", 51)
# QFI_gamma_X(3)
# QFI_gamma_X(4)
# QFI_gamma_X(6)

# QFI_at_K_T_zero(0.6, 0.6, 1.0, np.array([[0.5,0.5,0]]), np.ones(4) * np.pi, 15)
# QFI_at_K_T_zero(62/63, 1, 11/63, np.array([[0,0,0]]),15)

# QFI_at_K_T_zero(-0.09, -0.09, 1, np.array([[0,0,2*np.pi]]),15)
# QFI_at_K_T_zero(-0.09, -0.09, 1, np.array([[0,0,0]]),15)
# QFI_at_K_T_zero(-0.1, -0.1, 1, np.array([[0,0,2*np.pi]]),15)
# QFI_at_K_T_zero(-0.1, -0.1, 1, np.array([[0,0,0]]),15)
# QFI_at_K_T_zero(-0.06, -0.06, 1, np.array([[0,0,2*np.pi]]),15)
# QFI_at_K_T_zero(-0.06, -0.06, 1, np.array([[0,0,0]]),15)

def generate_K_points_pengcheng_dai(H_range_min, H_range_max, nH, K_range_min, K_range_max, nK, L_range_min, L_range_max, nL):
    """
    Generate a 3D grid where each point is a linear combination of H_vector, K_vector, and L_vector
    with coefficients spanning [-H_range, H_range], [-K_range, K_range], [-L_range, L_range].

    Parameters:
    -----------
    H_range, K_range, L_range : int
        Range values for coefficients
    
    Returns:
    --------
    K_points : ndarray
        Array of shape (n_points, 3) containing all K points
    """
    H_vector_real = 2*np.pi*np.array([1, 1, -2])
    K_vector_real = 2*np.pi*np.array([1, -1, 0])
    L_vector_real = 2*np.pi*np.array([1, 1, 1])

    H_vector = np.array([-0.5, -0.5, 1])
    K_vector = np.array([-0.5, 0.5, 0])
    L_vector = np.array([1, 1, 1])
    
    # Create coefficient ranges
    h_values = np.linspace(H_range_min, H_range_max, nH)
    k_values = np.linspace(K_range_min, K_range_max, nK)
    l_values = np.linspace(L_range_min, L_range_max, nL)

    # Create a grid of all possible combinations
    h_grid, k_grid, l_grid = np.meshgrid(h_values, k_values, l_values, indexing='ij')
    h_grid = h_grid.flatten()
    k_grid = k_grid.flatten()
    l_grid = l_grid.flatten()
    
    # Calculate K points using linear combinations
    K_points = np.zeros((len(h_grid), 3))
    for i in range(len(h_grid)):
        K_points[i] = h_grid[i] * H_vector + k_grid[i] * K_vector + l_grid[i] * L_vector
    # Calculate the volume element dV
    dV = np.abs(np.linalg.det(np.array([H_vector_real, K_vector_real, L_vector_real]))) / (nH * nK * nL)

    return K_points, dV

def PC_stuff(B):
    h_min, h_max, nH, k_min, k_max, nK, l_min, l_max, nL = -0.1, 0.1, 3, 0.739, 0.839, 1, -0.1, 0.1, 3

    int_grid, dV = generate_K_points_pengcheng_dai(h_min, h_max, nH, k_min, k_max, nK, l_min, l_max, nL)

    Jxx =  0.98412698412
    Jyy = 1.0 
    Jzz = 0.1746031746

    py = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=25, h=B, n=h111, flux=np.ones(4)*np.pi)
    py.solvemeanfield()

    omega = np.linspace(0, 10, 500)

    Szz, Szzglobal, Sxx, Sxxglobal = DSSF_int(int_grid, omega, py, 1e-6, dV=dV)
    
    # Save the data to text files
    np.savetxt(f"DSSF_B={B}_Szz_local.txt", np.column_stack((omega, Szz)), header='Energy (meV) Local Szz', fmt='%.6f')
    np.savetxt(f"DSSF_B={B}_Szz_global.txt", np.column_stack((omega, Szzglobal)), header='Energy (meV) Global Szz', fmt='%.6f')
    np.savetxt(f"DSSF_B={B}_Sxx_local.txt", np.column_stack((omega, Sxx)), header='Energy (meV) Local Sxx', fmt='%.6f')
    np.savetxt(f"DSSF_B={B}_Sxx_global.txt", np.column_stack((omega, Sxxglobal)), header='Energy (meV) Global Sxx', fmt='%.6f')
    
# PC_stuff(0)
# PC_stuff(0.1)
# PC_stuff(0.2)
# PC_stuff(0.3)
# PC_stuff(0.4)
# PC_stuff(0.5)
# PC_stuff(0.6)
# PC_stuff(0.7)
# PC_stuff(0.8)
# PC_stuff(0.9)
# PC_stuff(1.0)
