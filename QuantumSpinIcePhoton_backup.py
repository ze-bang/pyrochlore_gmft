import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import itertools
import time


plt.rcParams["text.usetex"]=True
plt.rcParams["font.family"]="serif"
plt.rcParams["font.serif"]=["CMU Serif"]
plt.rcParams["font.size"]=16

class QuantumSpinIcePhoton(object):
    #Define basis vectors in the GCC
    e0GCC = np.array([0, 0, 0])
    e1GCC = 1/2*np.array([0, 1, 1])
    e2GCC = 1/2*np.array([1, 0, 1])
    e3GCC = 1/2*np.array([1, 1, 0])
    eGCCVec = np.array([e0GCC, e1GCC, e2GCC, e3GCC])
    #Define NN translation vectors in the GCC
    b0GCC = 1/4*np.array([1, 1, 1])
    b1GCC = -1/4*np.array([-1, 1, 1])
    b2GCC = -1/4*np.array([1, -1, 1])
    b3GCC = -1/4*np.array([1, 1, -1])
    bGCCVec = np.array([b0GCC, b1GCC, b2GCC, b3GCC])
    h_mu_nu = 1/np.sqrt(8) * np.array([
        [np.zeros(3), np.cross(b0GCC,b1GCC)/np.linalg.norm(np.cross(b0GCC,b1GCC)), np.cross(b0GCC,b2GCC)/np.linalg.norm(np.cross(b0GCC,b2GCC)), np.cross(b0GCC,b3GCC)/np.linalg.norm(np.cross(b0GCC,b3GCC))],
        [np.cross(b1GCC,b0GCC)/np.linalg.norm(np.cross(b1GCC,b0GCC)), np.zeros(3), np.cross(b1GCC,b2GCC)/np.linalg.norm(np.cross(b1GCC,b2GCC)), np.cross(b1GCC,b3GCC)/np.linalg.norm(np.cross(b1GCC,b3GCC))],
        [np.cross(b2GCC,b0GCC)/np.linalg.norm(np.cross(b2GCC,b0GCC)), np.cross(b2GCC,b1GCC)/np.linalg.norm(np.cross(b2GCC,b1GCC)), np.zeros(3), np.cross(b2GCC,b3GCC)/np.linalg.norm(np.cross(b2GCC,b3GCC))],
        [np.cross(b3GCC,b0GCC)/np.linalg.norm(np.cross(b3GCC,b0GCC)), np.cross(b3GCC,b1GCC)/np.linalg.norm(np.cross(b3GCC,b1GCC)), np.cross(b3GCC,b2GCC)/np.linalg.norm(np.cross(b3GCC,b2GCC)), np.zeros(3)]
    ])
    #Define reciprocal lattice vectors
    Vuc = np.dot(e1GCC,np.cross(e2GCC, e3GCC))
    A1 = (2*np.pi/Vuc)*np.cross(e2GCC, e3GCC)
    A2 = (2*np.pi/Vuc)*np.cross(e3GCC, e1GCC)
    A3 = (2*np.pi/Vuc)*np.cross(e1GCC, e2GCC)
    VBZ = np.dot(A1,np.cross(A2, A3))
    # Define sublattice-dependant local frame
    # z-axis
    z0 = 1/np.sqrt(3)*np.array([1,1,1])
    z1 = 1/np.sqrt(3)*np.array([1,-1,-1])
    z2 = 1/np.sqrt(3)*np.array([-1,1,-1])
    z3 = 1/np.sqrt(3)*np.array([-1,-1,1])
    z_local_gcc_vec = np.array([z0, z1, z2, z3])
    # y-axis
    y0 = 1/np.sqrt(2)*np.array([0,-1,1])
    y1 = 1/np.sqrt(2)*np.array([0,1,-1])
    y2 = 1/np.sqrt(2)*np.array([0,-1,-1])
    y3 = 1/np.sqrt(2)*np.array([0,1,1])
    y_local_gcc_vec = np.array([y0, y1, y2, y3])
    # x-axis
    x0 = 1/np.sqrt(6)*np.array([-2,1,1])
    x1 = 1/np.sqrt(6)*np.array([-2,-1,-1])
    x2 = 1/np.sqrt(6)*np.array([2,1,-1])
    x3 = 1/np.sqrt(6)*np.array([2,-1,1])
    x_local_gcc_vec = np.array([x0, x1, x2, x3])
    # Define high symmetry points in the first Brillouin zone
    FBZ_Gamma_GCC = np.array([0,0,0])
    FBZ_X_GCC = 1/2*A1 + 1/2*A2
    #FBZ_X_GCC = 1/2*A2 + 1/2*A3
    FBZ_L_GCC = 1/2*A1 + 1/2*A2 + 1/2*A3
    FBZ_W_GCC = 1/4*A1 + 3/4*A2 + 1/2*A3
    FBZ_U_GCC = 1/4*A1 + 5/8*A2 + 5/8*A3
    #FBZ_K_GCC = 3/8*A1 + 3/4*A2 + 3/8*A3
    FBZ_K_GCC = 3/8*A1 + 3/8*A2 + 3/4*A3

    # Dictionnary to get latex symbol for each point
    symbol_points = {
        tuple(FBZ_Gamma_GCC) : r"$\Gamma$",
        tuple(FBZ_X_GCC) : r"X",
        tuple(FBZ_L_GCC) : r"L",
        tuple(FBZ_W_GCC) : r"W",
        tuple(FBZ_U_GCC) : r"U",
        tuple(FBZ_K_GCC) : r"K",
    }

    # Speed of light
    c_moessner = 0.51 #(a g/\hbar) from: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.117205
    c_shannon = 0.6 #(a g/\hbar) from: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.067204
    c_castelnovo = 0.41 #(a g/\hbar) from: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.134439

    def __init__(self, speed_of_light=None, W=0, kappa=1, temperature=0.0, renorm_param=1, Jpm=0, Jzz=1, B=0):
        self.Jpm = Jpm # Jpm coupling in units of Jzz
        self.Jzz = Jzz
        self.B = B
        self.temperature = temperature # Temperature in units of Jzz
        # Define coupling and gMFT ansätz
        self.speed_of_light = speed_of_light
        self.g0 = 24 * np.abs(self.Jpm**3)/self.Jzz**2
        self.Kappa = renorm_param * self.g0 # Following Benton's convention
        if speed_of_light==None:
           #self.speed_of_light = (self.c_moessner * self.g0)/2
           self.speed_of_light = (self.c_moessner * self.Kappa)/2
        self.U = self.speed_of_light**2/(self.Kappa) # Following Benton's convention
        self.W = W
        self.kappa = kappa
        print("QuantumSpinIcePhoton initialized with:")
        print(f"  - U = {self.U} (in units of Jzz)")
        print(f"  - W = {self.g0} (in units of Jzz)")
        print(f"  - Kappa = {self.Kappa} (in units of Jzz)")
        print(f"  - kappa = {self.kappa} (in units of Jzz)")
        print(f"  - Temperature = {self.temperature} (in units of Jzz)")
        print(f"  - Speed of light = {self.speed_of_light} (in units of a g/\hbar)")
        print(f"  - Renormation parameter = {renorm_param}")
    ##################
    # Useful functions
    def flatten_tuple(self, tupex):
        return tuple(itertools.chain.from_iterable(tupex))

    def flatten_list(self, listex):
        return list(itertools.chain.from_iterable(listex))

    def get_hhl_plane_gcc(self, n_h=11, n_l=11, min_h=-2, max_h=2, min_l=-3, max_l=3):
        return np.array([[[h,h,l] for h in np.linspace(min_h,max_h,n_h)] for l in np.linspace(min_l,max_l,n_l)]) #idx: l,h

    def interpolate_points(self, point1, point2, t):
        return (1-t)*point1 + t*point2

    def interpolate_points_vec(self, point1, point2, t_vec):
        return [(1-t)*point1 + t*point2 for t in t_vec]

    def define_path_FBZ(self, points, n):
        return [self.interpolate_points_vec(p1, p2, np.linspace(0, 1, int(np.floor(n*np.linalg.norm(p2-p1))))) for p1,p2 in zip(points[:-1],points[1:])]

    ############################
    # Save object in binary file
    def save_object(self, filename=None):
        if filename==None:
            filename = f"photon_calculation_U_{self.U:.5f}_W_{self.W:.5f}_Kappa_{self.Kappa:.5f}_kappa_{self.kappa:.5f}_temperature_{self.temperature:.5f}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    ##############################################
    # Compute the dispersion along a specific path
    def zeta_function_at_q_all_steps(self, q_vec):
        zeta = np.sqrt(2)*np.sum(np.sin(np.einsum('ijk,k->ij', self.h_mu_nu, q_vec))**2)
        return zeta

    def zeta_function_at_q(self, q_vec):
        zeta = 2 * np.sqrt(3 - np.cos(q_vec[0]/2)*np.cos(q_vec[1]/2) - np.cos(q_vec[0]/2)*np.cos(q_vec[2]/2) - np.cos(q_vec[1]/2)*np.cos(q_vec[2]/2))
        return zeta

    def compute_dispersion(self, q_vec):
        zeta = self.zeta_function_at_q(q_vec)
        return np.sqrt(self.U*self.Kappa*zeta**2 + self.W*self.Kappa*zeta**4)

    def compute_dispersion_along_path_from_points(self, points, n):
        path = self.define_path_FBZ(points, n)
        return [np.array([self.compute_dispersion(k)  for k in line]) for line in path]


    #############################################
    # Compute Static Spin Structure Factor (SSSF)
    # i) SSSF all components
    def _get_sssf_all_mu_nu_components_at_q(self, q_vec):
        sin_of_q_dot_h_mu_nu = np.sin(np.einsum('ijk,k->ij', self.h_mu_nu, q_vec)) # idx: mu,nu
        omega_at_q = self.compute_dispersion(q_vec) + 10**-7
        if self.temperature != 0:
            n_bose_at_q = 1/(np.exp(omega_at_q/self.temperature) - 1)
            S_mu_nu = self.kappa**2 * self.Kappa/(2 * omega_at_q) * np.einsum('ij,kj->ik', sin_of_q_dot_h_mu_nu, sin_of_q_dot_h_mu_nu) * (1 + 2 * n_bose_at_q)
        else:
            S_mu_nu = self.kappa**2 * self.Kappa/(2 * omega_at_q) * np.einsum('ij,kj->ik', sin_of_q_dot_h_mu_nu, sin_of_q_dot_h_mu_nu)
        if not hasattr(self, "num_sssf_points"):
            self.num_sssf_points = 0
        self.num_sssf_points += 1   # Define high symmetry points in the first Brillouin zone
        return S_mu_nu

    def _get_sssf_all_mu_nu_components_q_array(self, q_array):
        sssf_xy_mu_nu_q_array = np.array([self._get_sssf_all_mu_nu_components_at_q(q) for q in q_array]) #idx: q,mu,nu
        if not hasattr(self, "sssf_mu_nu_q_array"):
            self.sssf_mu_nu_q_array = {}
            self.sssf_calculation_info = {}
        point1 = tuple(q_array[0])
        point2 = tuple(q_array[-1])
        self.sssf_mu_nu_q_array[(point1,point2)] = sssf_xy_mu_nu_q_array
        self.sssf_calculation_info[(point1,point2)] = {
            "q_array":q_array,
        }

    def get_sssf_equal_time_correlation_local_frame_q_array(self, q_array):
        # Get the SSSF components (both the sublattice component (mu and nu) and the one associated wit the xx,xy,yx and yy components)
        point1 = tuple(q_array[0])
        point2 = tuple(q_array[-1])
        if not hasattr(self, "sssf_mu_nu_q_array"):
            self._get_sssf_all_mu_nu_components_q_array(q_array) #idx: q,mu,nu
        else:
            if (point1,point2) not in self.sssf_mu_nu_q_array:
                self._get_sssf_all_mu_nu_components_q_array(q_array) #idx: q,mu,nu
        correlation_local_frame = np.einsum('ijk->i', self.sssf_mu_nu_q_array[(point1,point2)])
        return correlation_local_frame

    def get_sssf_equal_time_scattering_from_components_q_array(self, q_array, x_basis=None):
        point1 = tuple(q_array[0])
        point2 = tuple(q_array[-1])
        if x_basis is None:
            x_basis = self.z_local_gcc_vec #idx: mu,i
        if not hasattr(self, "sssf_mu_nu_q_array"):
            self._get_sssf_all_mu_nu_components_q_array(q_array) #idx: qmu,nu
        else:
            if (point1,point2) not in self.sssf_mu_nu_q_array:
                self._get_sssf_all_mu_nu_components_q_array(q_array) #idx: q,x,y,mu,nu
        # Get the transverse projector for each component (xx,xy,yx,yy)
        normalized_q = np.array([
            q/np.linalg.norm(q) if np.linalg.norm(q)!=0 else np.array([0,0,0]) for q in q_array
        ])
        x_dot_q_array = np.einsum('ij,kj->ik', normalized_q, x_basis) # idx: q,mu
        id_matrix = np.ones(np.shape(q_array)[0])
        transverse_projector_part_1 = np.einsum('ij,kj,l->lik', x_basis, x_basis, id_matrix) # idx: q,mu,nu
        transverse_projector_part_2 = np.einsum('ij,ik->ijk', x_dot_q_array, x_dot_q_array)#/norm_part_2[:,None,None]
        transverse_projector = transverse_projector_part_1 - transverse_projector_part_2 #idx: q,mu,nu
        equal_time_scattering = np.einsum('ijk,ijk->i', transverse_projector, self.sssf_mu_nu_q_array[(point1,point2)]) #idx: q
        return equal_time_scattering

    def get_sssf_spin_flip_z_scattering_from_components_q_array(self, q_array, x_basis=None):
        point1 = tuple(q_array[0])
        point2 = tuple(q_array[-1])
        # Define local basis
        if x_basis is None:
            x_basis = self.z_local_gcc_vec #idx: mu,i
        # Get the SSSF compoenent (both the sublattice component (mu and nu) and the one associated wit the xx,xy,yx and yy components)
        if not hasattr(self, "sssf_mu_nu_q_array"):
            self._get_sssf_all_mu_nu_components_q_array(q_array) #idx: q,mu,nu
        else:
            if (point1,point2) not in self.sssf_mu_nu_q_array:
                self._get_sssf_all_mu_nu_components_q_array(q_array) #idx: q,mu,nu
        # Vector perpendicular to the scattering hhl plane
        z_scattering = np.array([1,-1,0])/np.sqrt(2)
        # Prefactor
        q_cross_z_scattering = np.array([ np.cross(q,z_scattering)/np.linalg.norm(np.cross(q,z_scattering)) for q in q_array]) #idx: q,i
        x_dot_q_cross_z_scattering = np.einsum('ij,kj->ki', x_basis, q_cross_z_scattering) #idx: q,mu
        prefactor = np.einsum('ij,ik->ijk', x_dot_q_cross_z_scattering, x_dot_q_cross_z_scattering)  #idx: q,mu,nu
        # Compute spin-flip equal-time scattering
        spin_flip_equal_time_scattering = np.einsum('ijk,ijk->i', prefactor, self.sssf_mu_nu_q_array[(point1,point2)]) # idx: q
        #return np.real(spin_flip_equal_time_scattering)
        return spin_flip_equal_time_scattering

    def get_sssf_non_spin_flip_z_scattering_from_components_q_array(self, q_array, x_basis=None):
        point1 = tuple(q_array[0])
        point2 = tuple(q_array[-1])
        # Define local basis
        if x_basis is None:
            x_basis = self.z_local_gcc_vec #idx: mu,i
        # Get the SSSF compoenent (both the sublattice component (mu and nu) and the one associated wit the xx,xy,yx and yy components)
        if not hasattr(self, "sssf_mu_nu_q_array"):
            self._get_sssf_all_mu_nu_components_q_array(q_array) #idx: q,x,y,mu,nu
        else:
            if (point1,point2) not in self.sssf_mu_nu_q_array:
                self._get_sssf_all_mu_nu_components_q_array(q_array) #idx: q,x,y,mu,nu
        # Vector perpendicular to the scattering hhl plane
        z_scattering = np.array([1,-1,0])/np.sqrt(2)
        # Prefactor
        x_dot_z_scattering = np.einsum('ij,j->i', x_basis, z_scattering) #idx: mu
        prefactor = np.einsum('i,j->ij', x_dot_z_scattering, x_dot_z_scattering) #idx: mu,nu
        # Compute non-spin-flip equal-time scattering
        non_spin_flip_equal_time_scattering = np.einsum('ij,kij->k', prefactor, self.sssf_mu_nu_q_array[(point1,point2)]) #idx: q
        return non_spin_flip_equal_time_scattering

    def compute_any_static_correlation_along_path(self, static_correlation_func, points, n=30, x_basis=None):
        path = self.define_path_FBZ(points, n)
        def get_sssf_and_catch_exception(func, q:np.ndarray):
            try:
                return func(q, x_basis=x_basis)
            except TypeError:
                return func(q)
        static_correlation_along_path = [
            get_sssf_and_catch_exception(static_correlation_func, np.array(q_array)) for q_array in path
        ]
        return static_correlation_along_path #idx: line,q

    def compute_any_static_correlation_along_any_path(self, static_correlation_func, points, x_basis=None):
        print(points)
        def get_sssf_and_catch_exception(func, q:np.ndarray):
            try:
                return func(q, x_basis=x_basis)
            except TypeError:
                return func(q)
        static_correlation_along_path = [
            get_sssf_and_catch_exception(static_correlation_func, points)
        ]
        return static_correlation_along_path #idx: line,q

    def compute_any_static_correlation_in_hhl_plane(self, static_correlation_func, n_h=11, n_l=11, min_h=-2, max_h=2, min_l=-3, max_l=3, x_basis=None):
        hhl_plane = self.get_hhl_plane_gcc(n_h=n_h, n_l=n_l, min_h=min_h, max_h=max_h, min_l=min_l, max_l=max_l)
        q_array = np.reshape(hhl_plane, (n_h*n_l,3))
        print(q_array.shape)
        print(q_array)

        try:
            static_correlation_hhl = static_correlation_func(q_array, x_basis=x_basis)
        except TypeError:
            static_correlation_hhl = static_correlation_func(q_array)
        static_correlation_hhl = np.reshape(static_correlation_hhl, (n_l,n_h))
        return static_correlation_hhl

    ################################################
    # Compute Dynamical Spin Structure Factor (DSSF)
    # ii) DSSF for the pi-flux state
    def _get_dssf_all_mu_nu_components_at_q(self, q_vec, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=100, eta=0.01):
        #print(f"{eta:.5f}")
        sin_of_q_dot_h_mu_nu = np.sin(np.einsum('ijk,k->ij', self.h_mu_nu, q_vec)) # idx: mu,nu
        omega_at_q = self.compute_dispersion(q_vec) + 10**-15
        omega_vec = np.linspace(omega_min_goal, omega_max_goal, N_OMEGA_GOAL)
        delta_function_min = 1/np.pi * (eta/2)/((omega_vec-omega_at_q)**2 + (eta/2)**2) # omega
        delta_function_plus = 1/np.pi * (eta/2)/((omega_vec+omega_at_q)**2 + (eta/2)**2) # omega
        if self.temperature != 0:
            n_bose_at_q = 1/(np.exp(omega_at_q/self.temperature) - 1)
            energy_factor = (((1 + n_bose_at_q) * delta_function_min) + n_bose_at_q * delta_function_plus)
            S_mu_nu = self.kappa**2 * self.Kappa/(2 * omega_at_q) * np.einsum('ij,kj,l->ikl', sin_of_q_dot_h_mu_nu, sin_of_q_dot_h_mu_nu, energy_factor)
        else:
            energy_factor = delta_function_min
            S_mu_nu = self.kappa**2 * self.Kappa/(2 * omega_at_q) * np.einsum('ij,kj,l->ikl', sin_of_q_dot_h_mu_nu, sin_of_q_dot_h_mu_nu, energy_factor)
        if not hasattr(self, "num_dssf_points"):
            self.num_dssf_points = 0
        self.num_dssf_points += 1   # Define high symmetry points in the first Brillouin zone
        return S_mu_nu

    def _get_dssf_all_mu_nu_components_q_array(self, q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=100, eta=0.01):
        # Create empty dictionnary to keep track of dssf along different lines
        if not hasattr(self, "dssf_xy_mu_nu_q_array"):
            self.dssf_mu_nu_q_array = {}
            self.dssf_calculation_info = {}
        # Compute DSSF
        dssf_mu_nu_q_array = np.array([
            self._get_dssf_all_mu_nu_components_at_q(q, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta) for q in q_array
        ]) #idx: q,mu,nu,omega
        # Save the DSSF componenst in a dictionnary so that I can reuse it for different calculations
        point1 = tuple(q_array[0])
        point2 = tuple(q_array[-1])
        self.dssf_mu_nu_q_array[(point1,point2)] = dssf_mu_nu_q_array
        self.dssf_calculation_info[(point1,point2)] = {
            "q_array":q_array,
            "omega_min":omega_min_goal,
            "omega_max":omega_max_goal,
            "N_OMEGA":N_OMEGA_GOAL,
            "eta":eta,
        }

    def get_dssf_correlation_local_frame_q_array(self, q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=100, eta=0.01):
        # Get the SSSF compoenent (both the sublattice component (mu and nu) and the one associated wit the xx,xy,yx and yy components)
        point1 = tuple(q_array[0])
        point2 = tuple(q_array[-1])
        if not hasattr(self, "dssf_xy_mu_nu_q_array"):
            self._get_dssf_all_mu_nu_components_q_array(q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta) #idx: q,mu,nu,omega
        else:
            if (point1,point2) not in self.dssf_mu_nu_q_array:
                self._get_dssf_all_mu_nu_components_q_array(q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta) #idx: q,mu,nu,omega
        correlation_local_frame = np.einsum('ijkl->il', self.dssf_mu_nu_q_array[(point1,point2)]) #idx: q,omega
        return np.real(correlation_local_frame)

    def get_dssf_total_scattering_from_components_q_array(self, q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=100, eta=0.01, x_basis=None):
        if x_basis is None:
            x_basis = self.z_local_gcc_vec #idx: mu,i
        # Get the SSSF compoenent (both the sublattice component (mu and nu) and the one associated wit the xx,xy,yx and yy components)
        point1 = tuple(q_array[0])
        point2 = tuple(q_array[-1])
        if not hasattr(self, "dssf_mu_nu_q_array"):
            self._get_dssf_all_mu_nu_components_q_array(q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta) #idx: q,x,y,mu,nu,omega
        else:
            if (point1,point2) not in self.dssf_mu_nu_q_array:
                self._get_dssf_all_mu_nu_components_q_array(q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta) #idx: q,x,y,mu,nu,omega
        # Get the transverse projector for each component
        normalized_q = np.array([
            q/np.linalg.norm(q) if np.linalg.norm(q)!=0 else np.array([0,0,0]) for q in q_array
        ])
        x_dot_q_array = np.einsum('ij,kj->ik', normalized_q, x_basis) # idx: q,mu
        id_matrix = np.ones(np.shape(q_array)[0])
        transverse_projector_part_1 = np.einsum('ij,kj,l->lik', x_basis, x_basis, id_matrix) # idx: q,mu,nu
        transverse_projector_part_2 = np.einsum('ij,ik->ijk', x_dot_q_array, x_dot_q_array)#/norm_part_2[:,None,None]
        transverse_projector = transverse_projector_part_1 - transverse_projector_part_2 #idx: q,mu,nu
        dynamical_scattering = np.einsum('ijk,ijkl->il', transverse_projector, self.dssf_mu_nu_q_array[(point1,point2)]) #idx: q
        return dynamical_scattering

    def get_dssf_spin_flip_z_scattering_from_components_q_array(self, q_array,omega_min_goal, omega_max_goal, N_OMEGA_GOAL=100, eta=0.01, x_basis=None):
        # Define local basis
        if x_basis is None:
            x_basis = self.x_local_gcc_vec #idx: mu,i
        # Get the DSSF compoenent (both the sublattice component (mu and nu) and the one associated wit the xx,xy,yx and yy components)
        point1 = tuple(q_array[0])
        point2 = tuple(q_array[-1])
        if not hasattr(self, "dssf_mu_nu_q_array"):
            self._get_dssf_all_mu_nu_components_q_array(q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta) #idx: q,x,y,mu,nu,omega
        else:
            if (point1,point2) not in self.dssf_mu_nu_q_array:
                self._get_dssf_all_mu_nu_components_q_array(q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta) #idx: q,x,y,mu,nu,omega
        # Vector perpendicular to the scattering hhl plane
        z_scattering = np.array([1,-1,0])/np.sqrt(2)
        # Prefactor
        q_cross_z_scattering = np.array([
            np.cross(q,z_scattering)/np.linalg.norm(np.cross(q,z_scattering)) if np.linalg.norm(np.cross(q,z_scattering))!=0 else np.array([0,0,0])
            for q in q_array
        ]) #idx: q,i
        x_dot_q_cross_z_scattering = np.einsum('ij,kj->ki', x_basis, q_cross_z_scattering) #idx: q,mu
        prefactor = np.einsum('ij,ik->ijk', x_dot_q_cross_z_scattering, x_dot_q_cross_z_scattering)  #idx: q,mu,nu
        # Compute spin-flip equal-time scattering
        spin_flip_dynamical_scattering = np.einsum('ijk,ijkl->il', prefactor, self.dssf_mu_nu_q_array[(point1,point2)]) # idx: q
        #return np.real(spin_flip_equal_time_scattering)
        return spin_flip_dynamical_scattering

    def get_dssf_non_spin_flip_z_scattering_from_components_q_array(self, q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=100, eta=0.01, x_basis=None):
        # Define local basis
        if x_basis is None:
            x_basis = self.z_local_gcc_vec #idx: mu,i
        # Get the DSSF compoenent (both the sublattice component (mu and nu) and the one associated wit the xx,xy,yx and yy components)
        point1 = tuple(q_array[0])
        point2 = tuple(q_array[-1])
        if not hasattr(self, "dssf_xy_mu_nu_q_array"):
            self._get_dssf_all_mu_nu_components_q_array(q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta) #idx: q,x,y,mu,nu,omega
        else:
            if (point1,point2) not in self.dssf_xy_mu_nu_q_array:
                self._get_dssf_all_mu_nu_components_q_array(q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta) #idx: q,x,y,mu,nu,omega
        # Vector perpendicular to the scattering hhl plane
        z_scattering = np.array([1,-1,0])/np.sqrt(2)
        # Prefactor
        x_dot_z_scattering = np.einsum('ij,j->i', x_basis, z_scattering) #idx: mu
        prefactor = np.einsum('i,j->ij', x_dot_z_scattering, x_dot_z_scattering) #idx: mu,nu
        # Compute non-spin-flip equal-time scattering
        non_spin_flip_equal_time_scattering = np.einsum('ij,kijl->kl', prefactor, self.dssf_mu_nu_q_array[(point1,point2)]) #idx: q
        return non_spin_flip_equal_time_scattering

    def compute_any_dynamical_correlation_along_path(self, dynamical_correlation_func, points, omega_min_goal, omega_max_goal,
                                                     N_OMEGA_GOAL=100, eta=0.01, n=40, x_basis=None):
        path = self.define_path_FBZ(points, n)
        def get_dssf_and_catch_exception(func, q:np.ndarray):
            try:
                return func(
                    q, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta, x_basis=x_basis
                )
            except TypeError:
                return func(q, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta)

        dynamical_correlation_along_path = [
            get_dssf_and_catch_exception(dynamical_correlation_func, np.array(q_array)) for q_array in path
        ]
        return dynamical_correlation_along_path #idx: line,q,omega

    def compute_any_dynamical_correlation_along_any_path(self, dynamical_correlation_func, points, omega_min_goal, omega_max_goal,
                                                     N_OMEGA_GOAL=100, eta=0.01, n=40, x_basis=None):
        def get_dssf_and_catch_exception(func, q:np.ndarray):
            try:
                return func(
                    q, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta, x_basis=x_basis
                )
            except TypeError:
                return func(q, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta)

        dynamical_correlation_along_path = [
            get_dssf_and_catch_exception(dynamical_correlation_func, points)
        ]
        return dynamical_correlation_along_path #idx: line,q,omega

    def compute_any_dynamical_correlation_in_hhl_plane(self, dynamical_correlation_func, omega_min_goal, omega_max_goal, n_h=11, n_l=11, min_h=-2, max_h=2,
                                                       min_l=-3, max_l=3, N_OMEGA_GOAL=100, eta=0.01, x_basis=None):
        hhl_plane = self.get_hhl_plane_gcc(n_h=n_h, n_l=n_l, min_h=min_h, max_h=max_h, min_l=min_l, max_l=max_l)
        q_array = np.reshape(hhl_plane, (n_h*n_l,3))
        try:
            dynamical_correlation_hhl = dynamical_correlation_func(
                q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta, x_basis=x_basis
            )
        except TypeError:
            dynamical_correlation_hhl = dynamical_correlation_func(q_array, omega_min_goal, omega_max_goal, N_OMEGA_GOAL=N_OMEGA_GOAL, eta=eta)
        # dynamical_correlation_hhl = np.reshape(dynamical_correlation_hhl, (n_l, n_h, N_OMEGA_GOAL))
        return dynamical_correlation_hhl

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
    H_vector = np.array([1, 1, -2])
    K_vector = np.array([1, -1, 0])
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
    
    return K_points


def get_hh2k_knk(H_range_min, H_range_max, nH, K_range_min, K_range_max, nK):
    H_vector = np.array([1, 1, -2])
    K_vector = np.array([1, -1, 0])    
    # Create coefficient ranges
    h_values = np.linspace(H_range_min, H_range_max, nH)
    k_values = np.linspace(K_range_min, K_range_max, nK)

    # Create a grid of all possible combinations
    h_grid, k_grid = np.meshgrid(h_values, k_values)
    h_grid = h_grid.flatten()
    k_grid = k_grid.flatten()
    
    # Calculate K points using linear combinations
    K_points = np.zeros((len(h_grid), 3))
    for i in range(len(h_grid)):
        K_points[i] = h_grid[i] * H_vector + k_grid[i] * K_vector

    return K_points


def dssf_integrated_K():
    # Generate grid of K points in the specified region
    nH, nK, nL = 51, 21, 51
    h_min, h_max = -0.1 * 2 * np.pi, 0.1 * 2 * np.pi
    k_min, k_max = 0.739 * 2 * np.pi, 0.839 * 2 * np.pi
    l_min, l_max = -0.1 * 2 * np.pi, 0.1 * 2 * np.pi
    int_grid = generate_K_points_pengcheng_dai(h_min, h_max, nH, k_min, k_max, nK, l_min, l_max, nL)
    n_omega = 500  # Number of omega points
    n_min = -0.2
    n_max = 0.3
    Jzz = 0.063
    Jpm = 0.1825
    speed_of_light = 0.0028
    B = 0.0
    ham = QuantumSpinIcePhoton(Jzz=Jzz, Jpm=Jpm, kappa=1, W=0, temperature=4.0*speed_of_light, speed_of_light=speed_of_light, B=B)
    local_frame = ham.compute_any_dynamical_correlation_along_any_path(
        ham.get_dssf_total_scattering_from_components_q_array, int_grid, n_min, n_max, n_omega
    )

    # Calculate volume element for L integration
    dL = 0.2 / (nL - 1)
    local_frame = np.array(local_frame)  # Convert to numpy array if not already
    reshaped_data = local_frame.reshape(nH, nK, nL, n_omega)
    
    # Integrate along L dimension but keep H, K, and omega
    L_integrated = np.sum(reshaped_data, axis=2) * dL 
    L_integrated = np.sum(L_integrated, axis=2) * (n_max - n_min) / n_omega  # Average over omega
    # Create meshgrid for H and omega values for plotting
    H_values = np.linspace(h_min, h_max, nH)
    K_values = np.linspace(k_min, k_max, nK)

    H_grid, K_grid = np.meshgrid(H_values, K_values)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(H_grid, K_grid, L_integrated.T, cmap='viridis', 
                          linewidth=0, antialiased=True)
    
    # Add labels and colorbar
    ax.set_xlabel('H')
    ax.set_ylabel('K')
    ax.set_zlabel('Intensity (integrated along L)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Save the plot
    plt.savefig('DSSF_H_omega_3D.pdf')
    plt.show()
        
    # Calculate volume element
    dH = 0.2 / (nH - 1)
    dK = 0.1 / (nK - 1)
    dL = 0.2 / (nL - 1)
    vol_element = dH * dK * dL
    
    # Reshape and perform integration
    # local_frame is a list of arrays, we need to concatenate them
    local_frame_array = np.concatenate(local_frame, axis=0)
    
    # Reshape to match grid dimensions [H, K, L, omega]
    reshaped_data = local_frame_array.reshape(nH, nK, nL, n_omega)
    
    # Perform integration over the K space for each omega
    integrated_spectrum = np.zeros(n_omega)
    for w in range(n_omega):
        integrated_spectrum[w] = np.sum(reshaped_data[:,:,:,w]) * vol_element
    
    # Plot the integrated spectrum
    omega_vals = np.linspace(n_min, n_max, n_omega)
    plt.figure(figsize=(10, 6))
    plt.plot(omega_vals, integrated_spectrum)
    plt.xlabel('Energy (meV)')
    plt.ylabel('Intensity (arb. units)')
    plt.title('Integrated Dynamical Structure Factor')
    plt.savefig('integrated_spectrum.pdf')
    plt.show()
    
    return integrated_spectrum

def sssf_under_111_field():
    min_h = -2 * 2 * np.pi
    max_h = 2 * 2 * np.pi
    min_l = -2 * 2 * np.pi
    max_l = 2 * 2 * np.pi
    n_h = 51
    n_l = 51
    k_grid = get_hh2k_knk(min_h, max_h, n_h, min_l, max_l, n_l)
    Jzz = 1
    Jpm = -0.3
    g0 = 24 * np.abs(Jpm**3)/Jzz**2
    Kappa = g0 # Following Benton's convention
    speed_of_light = (0.51 * Kappa)/2

    ham = QuantumSpinIcePhoton(Jzz, kappa=1, W=0, temperature=4.0*speed_of_light)
    local_frame = ham.compute_any_static_correlation_along_any_path(
        ham.get_sssf_equal_time_scattering_from_components_q_array, k_grid
    )
    local_frame = np.reshape(local_frame, (n_h, n_l))
    # local_frame = ham.compute_any_static_correlation_in_hhl_plane(
    #     ham.get_sssf_equal_time_scattering_from_components_q_array, n_h=51, n_l=51, min_h=min_h, max_h=max_h, min_l=min_l, max_l=max_l, x_basis=None
    # )

    f, ax = plt.subplots(figsize=(9,8))
    cf1 = ax.imshow(local_frame, extent=[min_h/(2*np.pi), max_h/(2*np.pi), min_l/(2*np.pi), max_l/(2*np.pi)], aspect=1/np.sqrt(2))
    f.colorbar(cf1, ax=ax)
    ax.set_xlabel("[hh-2h]")
    ax.set_ylabel("[k-k0]")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(width=1.2)
    plt.savefig("local_frame_sssf.pdf", format='pdf')
    plt.show()

def main():
    U = 1
    W = 0
    kappa = 1
    Kappa = 1

    n_h = 200
    n_l = 200
    min_h = -2 * 2 * np.pi
    max_h = 2 * 2 * np.pi
    min_l = -2 * 2 * np.pi
    max_l = 2 * 2 * np.pi
    h_vector = np.linspace(min_h, max_h, n_h)
    l_vector = np.linspace(min_l, max_l, n_l)

    Jzz = 1
    Jpm = 0.04

    g0 = 24 * np.abs(Jpm**3)/Jzz**2
    Kappa = g0 # Following Benton's convention
    speed_of_light = (0.51 * Kappa)/2

    ham = QuantumSpinIcePhoton(Jzz, kappa=1, W=0, temperature=4.0*speed_of_light)

    # Local Frame
    local_frame = ham.compute_any_static_correlation_in_hhl_plane(
        ham.get_sssf_equal_time_correlation_local_frame_q_array, n_h=n_h, n_l=n_l, min_h=min_h, max_h=max_h, min_l=min_l, max_l=max_l, x_basis=None
    )

    f, ax = plt.subplots(figsize=(9,8))
    cf1 = ax.imshow(local_frame, extent=[min_h, max_h, min_l, max_l], aspect=1/np.sqrt(2))
    f.colorbar(cf1, ax=ax)
    ax.set_xlabel("[hh0]")
    ax.set_ylabel("[00l]")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(width=1.2)
    plt.savefig("local_frame_sssf.pdf", format='pdf')
    plt.show()

    min_h = -2 * 2 * np.pi
    max_h = 2 * 2 * np.pi
    min_l = -3 * 2 * np.pi
    max_l = 3 * 2 * np.pi
    h_vector = np.linspace(min_h, max_h, n_h)
    l_vector = np.linspace(min_l, max_l, n_l)

    # Total
    total = ham.compute_any_static_correlation_in_hhl_plane(
        ham.get_sssf_equal_time_scattering_from_components_q_array, n_h=n_h, n_l=n_l, min_h=min_h, max_h=max_h, min_l=min_l, max_l=max_l, x_basis=None
    )
    f, ax = plt.subplots(figsize=(9,8))
    cf1 = ax.imshow(total, extent=[min_h, max_h, min_l, max_l], aspect=1/np.sqrt(2))
    f.colorbar(cf1, ax=ax)
    ax.set_xlabel("[hh0]")
    ax.set_ylabel("[00l]")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(width=1.2)
    plt.savefig("total_sssf.pdf", format='pdf')
    plt.show()

    # NSF
    non_spin_flip = ham.compute_any_static_correlation_in_hhl_plane(
        ham.get_sssf_non_spin_flip_z_scattering_from_components_q_array, n_h=n_h, n_l=n_l, min_h=min_h, max_h=max_h, min_l=min_l, max_l=max_l, x_basis=None
    )
    f, ax = plt.subplots(figsize=(9,8))
    cf1 = ax.imshow(non_spin_flip, extent=[min_h, max_h, min_l, max_l], aspect=1/np.sqrt(2))
    f.colorbar(cf1, ax=ax)
    ax.set_xlabel("[hh0]")
    ax.set_ylabel("[00l]")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(width=1.2)
    plt.savefig("non_spin_flip_sssf.pdf", format='pdf')
    plt.show()

    # SF
    spin_flip = ham.compute_any_static_correlation_in_hhl_plane(
        ham.get_sssf_spin_flip_z_scattering_from_components_q_array, n_h=n_h, n_l=n_l, min_h=min_h, max_h=max_h, min_l=min_l, max_l=max_l, x_basis=None
    )
    f, ax = plt.subplots(figsize=(9,8))
    cf1 = ax.imshow(spin_flip, extent=[min_h, max_h, min_l, max_l], aspect=1/np.sqrt(2))
    f.colorbar(cf1, ax=ax)
    ax.set_xlabel("[hh0]")
    ax.set_ylabel("[00l]")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(width=1.2)
    plt.savefig("spin_flip_sssf.pdf", format='pdf')
    plt.show()

def plot_dispersion_along_point(ham, points, n=40, show_plot=True, savefig=False, file_output_name="dispersion.pdf"):
    # Compute dispersion
    disp = ham.compute_dispersion_along_path_from_points(points, n)#/ham.speed_of_light
    # Creat figure
    f, ax = plt.subplots(figsize=(9,8))
    # Create x axis
    dist_vec = [ np.linalg.norm(k1-k2) for k1,k2 in zip(points[:-1],points[1:]) ]
    dist_vec = ham.flatten_list([[0],[d+np.sum(dist_vec[:i]) for i,d in enumerate(dist_vec) ]])
    x_vec = ham.flatten_list([np.linspace(d1,d2,len(disp[i][:])) for i,(d1,d2) in enumerate(zip(dist_vec[:-1],dist_vec[1:]))])
    # Set xticks
    my_xticks = [
        ham.symbol_points[tuple(pt)] if tuple(pt) in ham.symbol_points else f"[{pt[0]/(2*np.pi):.1f},{pt[1]/(2*np.pi):.1f},{pt[2]/(2*np.pi):.1f}]"
        for pt in points
    ]
    ax.set_xticks(dist_vec)
    ax.set_xticklabels(my_xticks)
    # Plot bands
    glued_disp = ham.flatten_list([disp[j][:] for j,p in enumerate(points[:-1])])
    ax.plot(x_vec, glued_disp, color='red', lw=1.5)
    # Aesthetic apsects
    for xc in dist_vec:
        ax.axvline(x=xc, color="black", linewidth=0.8, linestyle="dashdot", zorder=0)
    ax.set_ylabel(r"$\omega$ (u.a.)",fontsize=20)
    ax.set_xlim([x_vec[0], x_vec[-1]])
    ax.set_ylim(bottom=0.0,top=1.05*np.amax((ham.flatten_list(disp))))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
        ax.tick_params(width=1.2)
    plt.tight_layout()
    if savefig:
        plt.savefig(file_output_name, format='pdf')
    if show_plot:
        print("showing")
        plt.show()
    else:
        print("closing")
        plt.close()

def main_dispersion():
    #Jzz = 1
    #Jpm = 0.08
    #Jpm = -0.2
    #ham = QuantumSpinIcePhoton(Jzz, Jpm,renorm_param=1, kappa=1, W=0, temperature=0.0)
    temperature = 10
    speed_of_light = 0.0011
    ham = QuantumSpinIcePhoton(speed_of_light, kappa=1, W=0, temperature=temperature*speed_of_light)

    points = [
        np.array([-0.5, -0.5, -0.5])*2*np.pi, np.array([0, 0, 0])*2*np.pi, np.array([2,2,2])*2*np.pi, np.array([2,2,0])*2*np.pi, np.array([1,1,0])*2*np.pi, np.array([1,1,1])*2*np.pi
    ]
    points = [ham.FBZ_Gamma_GCC, ham.FBZ_X_GCC, ham.FBZ_W_GCC, ham.FBZ_K_GCC, ham.FBZ_L_GCC, ham.FBZ_Gamma_GCC]
    n = 100

    #disp = ham.compute_dispersion_along_path_from_points(points, n)
    print("plotting")
    plot_dispersion_along_point(ham, points)
    print("plotting")

def main_temperature():
    U = 1
    W = 0
    kappa = 1

    n_h = 50
    n_l = 50
    min_h_1 = -2.5 * 2 * np.pi
    max_h_1 = 2.5 * 2 * np.pi
    min_l_1 = -2.5 * 2 * np.pi
    max_l_1 = 2.5 * 2 * np.pi
    min_h = -2.5
    max_h = 2.5
    min_l = -2.5
    max_l = 2.5
    h_vector = np.linspace(min_h, max_h, n_h)
    l_vector = np.linspace(min_l, max_l, n_l)

    Jzz = 1
    Jpm = 0.04

    g0 = 24 * np.abs(Jpm**3)/Jzz**2
    Kappa = g0 # Following Benton's convention
    speed_of_light = (0.51 * Kappa)/2

    for temperature in [0, 0.1, 0.2, 0.3]:
        ham = QuantumSpinIcePhoton(1, kappa=2, W=0, temperature=temperature)

        total = ham.compute_any_dynamical_correlation_in_hhl_plane(ham.get_dssf_correlation_local_frame_q_array, 0, 6,
                                                                   n_h=n_h, n_l=n_l, min_h=min_h_1, max_h=max_h_1, min_l=min_l_1, max_l=max_l_1, N_OMEGA_GOAL=200)
        np.savetxt("SSSF_photon"+str(temperature)+".txt", total)
        # Total
        total = ham.compute_any_static_correlation_in_hhl_plane(
            ham.get_sssf_equal_time_scattering_from_components_q_array, n_h=n_h, n_l=n_l, min_h=min_h_1, max_h=max_h_1, min_l=min_l_1, max_l=max_l_1, x_basis=None
        )
        f, ax = plt.subplots(figsize=(9,8))
        cf1 = ax.imshow(total, extent=[min_h, max_h, min_l, max_l], aspect=1/np.sqrt(2))
        f.colorbar(cf1, ax=ax)
        ax.set_xlabel(r"$[hh0]$")
        ax.set_ylabel(r"$[00l]$")
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2)
            ax.tick_params(width=1.2)
        #ax.title.set_text(r"Total: $k_B T$={0:.2f}$\cdot h c/a_0$".format(temperature))
        f.savefig(f"total_temp_{temperature}.pdf", bbox_inches=None, pad_inches=0.0)
        plt.close()
        
        # NSF
        non_spin_flip = ham.compute_any_static_correlation_in_hhl_plane(
            ham.get_sssf_non_spin_flip_z_scattering_from_components_q_array, n_h=n_h, n_l=n_l, min_h=min_h_1, max_h=max_h_1, min_l=min_l_1, max_l=max_l_1, x_basis=None
        )
        f, ax = plt.subplots(figsize=(9,8))
        cf1 = ax.imshow(non_spin_flip, extent=[min_h, max_h, min_l, max_l], aspect=1/np.sqrt(2))
        f.colorbar(cf1, ax=ax)
        ax.set_xlabel(r"$[hh0]$")
        ax.set_ylabel(r"$[00l]$")
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2)
            ax.tick_params(width=1.2)
        #ax.title.set_text(r"NSF: $k_B T$={0:.2f}$\cdot h c/a_0$".format(temperature))
        f.savefig(f"nsf_temp_{temperature}.pdf", bbox_inches=None, pad_inches=0.0)
        plt.close()
        
        # SF
        spin_flip = ham.compute_any_static_correlation_in_hhl_plane(
            ham.get_sssf_spin_flip_z_scattering_from_components_q_array, n_h=n_h, n_l=n_l, min_h=min_h_1, max_h=max_h_1, min_l=min_l_1, max_l=max_l_1, x_basis=None
        )
        f, ax = plt.subplots(figsize=(9,8))
        cf1 = ax.imshow(spin_flip, extent=[min_h, max_h, min_l, max_l], aspect=1/np.sqrt(2))
        f.colorbar(cf1, ax=ax)
        ax.set_xlabel(r"$[hh0]$")
        ax.set_ylabel(r"$[00l]$")
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2)
            ax.tick_params(width=1.2)
        #ax.title.set_text(r"SF: $k_B T$={0:.2f}$\cdot h c/a_0$".format(temperature))
        f.savefig(f"sf_temp_{temperature}.pdf", bbox_inches=None, pad_inches=0.0)
        plt.close()
        
        # NSF - SF
        f, ax = plt.subplots(figsize=(9,8))
        cf1 = ax.imshow(non_spin_flip - spin_flip, extent=[min_h, max_h, min_l, max_l], aspect=1/np.sqrt(2))
        f.colorbar(cf1, ax=ax)
        ax.set_xlabel(r"$[hh0]$")
        ax.set_ylabel(r"$[00l]$")
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2)
            ax.tick_params(width=1.2)
        #ax.title.set_text(r"NSF-SF: $k_B T$={0:.2f}$\cdot h c/a_0$".format(temperature))
        f.savefig(f"diff_temp_{temperature}.pdf", bbox_inches=None, pad_inches=0.0)
        plt.close()


# Solving the self-consistency equations
if __name__=="__main__":
    # sssf_under_111_field()
    dssf_integrated_K()
