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
def QFI_gamma_X(n_sites):
    X = np.array([[0,0,2*np.pi]])

    n = 40
    Szz = np.zeros(n)
    Sxx = np.zeros(n)
    T = np.logspace(0, 4, n)
    T = 1 / T
    for i in range(n):
        print("Temp = ", T[i])
        tempzz, tempxx = SSSF_q_omega_beta_at_K(T[i], X, 500, -0.09, -0.09, 1, 0, h110, np.zeros(4), n_sites, 0)
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

    Gamma = np.array([[0,0,0]])

    Szz = np.zeros(n)
    Sxx = np.zeros(n)
    T = np.logspace(0, 4, n)
    T = 1 / T
    for i in range(n):
        print("Temp = ", T[i])
        tempzz, tempxx = SSSF_q_omega_beta_at_K(T[i], Gamma, 500, -0.09, -0.09, 1, 0, h110, np.zeros(4), n_sites, 0)
        Szz[i] = quantum_fisher_information_K(tempzz, T[i])[0]
        Sxx[i] = quantum_fisher_information_K(tempxx, T[i])[0]
    np.savetxt("QFI_T_Spm_local_Gamma_L={}.dat".format(n_sites), np.column_stack((T, Szz)), header="T/|J_{zz}|^{-1} F_{QFI}[S^{pm}]")
    np.savetxt("QFI_T_Spp_local_Gamma_L={}.dat".format(n_sites), np.column_stack((T, Sxx)), header="T/|J_{zz}|^{-1} F_{QFI}[S^{pp}]")

    # Szz = np.loadtxt("SSSF_0_flux_T=0_Szz_local_X.txt")
    # Sxx = np.loadtxt("SSSF_0_flux_T=0_Sxx_local_X.txt")
    plt.plot(T, Sxx, label="Spm")
    plt.plot(T, Szz, label="Spp")

    plt.legend([r"$F_{QFI}[S^{Spm}]$", r"$F_{QFI}[S^{Spp}]$"])
    plt.title(r"$F_{QFI}[S^{\alpha}]$  vs $T$ at $q=\Gamma$")
    plt.xlabel(r"$T/|J_{zz}|^{-1}$")
    plt.ylabel(r"$F_{QFI}$")
    plt.xscale("log")
    plt.savefig("SSSF_0_flux_T=0.pdf")

# QFI_gamma_X(3)
# QFI_gamma_X(4)
# QFI_gamma_X(6)
# QFI_gamma_X(25)
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
    h_min, h_max, nH, k_min, k_max, nK, l_min, l_max, nL = -0.1, 0.1, 5, 0.739, 0.839, 3, -0.1, 0.1, 5

    int_grid, dV = generate_K_points_pengcheng_dai(h_min, h_max, nH, k_min, k_max, nK, l_min, l_max, nL)

    Jxx =  0.98412698412
    Jyy = 1.0 
    Jzz = 0.1746031746

    py = pycon.piFluxSolver(Jxx, Jyy, Jzz, BZres=25, h=B, n=h111, flux=np.ones(4)*np.pi)
    py.solvemeanfield()

    omega = np.linspace(0, 10, 500)

    Szz, Szzglobal, Sxx, Sxxglobal = DSSF_int(int_grid, omega, py, 1e-6, dV=dV)
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot local Szz
    im1 = axes[0, 0].pcolormesh(omega, range(len(int_grid)), Szz, shading='auto')
    axes[0, 0].set_title('Local Szz')
    axes[0, 0].set_xlabel('Energy (meV)')
    axes[0, 0].set_ylabel('Q-point index')
    fig.colorbar(im1, ax=axes[0, 0])
    
    # Plot global Szz
    im2 = axes[0, 1].pcolormesh(omega, range(len(int_grid)), Szzglobal, shading='auto')
    axes[0, 1].set_title('Global Szz')
    axes[0, 1].set_xlabel('Energy (meV)')
    axes[0, 1].set_ylabel('Q-point index')
    fig.colorbar(im2, ax=axes[0, 1])
    
    # Plot local Sxx
    im3 = axes[1, 0].pcolormesh(omega, range(len(int_grid)), Sxx, shading='auto')
    axes[1, 0].set_title('Local Sxx')
    axes[1, 0].set_xlabel('Energy (meV)')
    axes[1, 0].set_ylabel('Q-point index')
    fig.colorbar(im3, ax=axes[1, 0])
    
    # Plot global Sxx
    im4 = axes[1, 1].pcolormesh(omega, range(len(int_grid)), Sxxglobal, shading='auto')
    axes[1, 1].set_title('Global Sxx')
    axes[1, 1].set_xlabel('Energy (meV)')
    axes[1, 1].set_ylabel('Q-point index')
    fig.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f"DSSF_B={B}_plots.pdf")

    # Save the data to text files
    np.savetxt(f"DSSF_B={B}_Szz_local.txt", np.column_stack((omega, Szz)), header='Energy (meV) Local Szz', fmt='%.6f')
    np.savetxt(f"DSSF_B={B}_Szz_global.txt", np.column_stack((omega, Szzglobal)), header='Energy (meV) Global Szz', fmt='%.6f')
    np.savetxt(f"DSSF_B={B}_Sxx_local.txt", np.column_stack((omega, Sxx)), header='Energy (meV) Local Sxx', fmt='%.6f')
    np.savetxt(f"DSSF_B={B}_Sxx_global.txt", np.column_stack((omega, Sxxglobal)), header='Energy (meV) Global Sxx', fmt='%.6f')
    
PC_stuff(0)
PC_stuff(0.1)
PC_stuff(0.2)
