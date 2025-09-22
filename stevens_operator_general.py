import numpy as np
from functools import lru_cache

class StevensOperators:
    """
    Class for generating Stevens operators O_l^m for any given J value.
    """
    
    def __init__(self, J):
        """
        Initialize Stevens operators for a given total angular momentum J.
        
        Parameters:
        -----------
        J : float
            Total angular momentum quantum number
        """
        self.J = J
        self.dim = int(2*J + 1)
        self.m_values = np.arange(-J, J+1)
        self.X = J*(J+1)
        
        # Pre-compute frequently used matrices
        self._Jz = None
        self._Jplus = None
        self._Jminus = None
        
    @property
    def Jz(self):
        """Jz operator matrix"""
        if self._Jz is None:
            self._Jz = np.diag(self.m_values)
        return self._Jz
    
    @property
    def Jplus(self):
        """J+ operator matrix"""
        if self._Jplus is None:
            self._Jplus = np.zeros((self.dim, self.dim), dtype=complex)
            for i in range(self.dim-1):
                m = self.m_values[i]
                self._Jplus[i+1, i] = np.sqrt(self.X - m*(m+1))
        return self._Jplus
    
    @property
    def Jminus(self):
        """J- operator matrix"""
        if self._Jminus is None:
            self._Jminus = self.Jplus.T.conj()
        return self._Jminus
    
    def _matrix_power(self, M, n):
        """Compute matrix power efficiently"""
        if n == 0:
            return np.eye(self.dim, dtype=complex)
        elif n == 1:
            return M
        else:
            return np.linalg.matrix_power(M, n)
    
    def _anticommutator(self, A, B):
        """Compute anticommutator {A,B} = AB + BA"""
        return A @ B + B @ A
    
    def get_operator(self, l, m):
        """
        Get Stevens operator O_l^m matrix.
        
        Parameters:
        -----------
        l : int
            Rank of the operator (2, 4, or 6)
        m : int
            Order of the operator (-l <= m <= l)
            
        Returns:
        --------
        numpy.ndarray
            Matrix representation of O_l^m
        """
        if abs(m) > l:
            raise ValueError(f"Invalid m={m} for l={l}. Must have |m| <= l")
            
        # Define operator functions
        operators = {
            (2, 0): self._O20,
            (2, 2): self._O22,
            (2, -2): self._O2m2,
            (4, 0): self._O40,
            (4, 2): self._O42,
            (4, -2): self._O4m2,
            (4, 4): self._O44,
            (4, -4): self._O4m4,
            (6, 0): self._O60,
            (6, 2): self._O62,
            (6, -2): self._O6m2,
            (6, 4): self._O64,
            (6, -4): self._O6m4,
            (6, 6): self._O66,
            (6, -6): self._O6m6,
        }
        
        if (l, m) not in operators:
            raise ValueError(f"Stevens operator O_{l}^{m} not implemented")
            
        return operators[(l, m)]()
    
    # Rank 2 operators
    def _O20(self):
        """O_2^0 = 3J_z^2 - J(J+1)"""
        return 3 * self._matrix_power(self.Jz, 2) - self.X * np.eye(self.dim)
    
    def _O22(self):
        """O_2^2 = (J_+^2 + J_-^2)/2"""
        Jp2 = self._matrix_power(self.Jplus, 2)
        Jm2 = self._matrix_power(self.Jminus, 2)
        return (Jp2 + Jm2) / 2
    
    def _O2m2(self):
        """O_2^{-2} = (J_+^2 - J_-^2)/(2i)"""
        Jp2 = self._matrix_power(self.Jplus, 2)
        Jm2 = self._matrix_power(self.Jminus, 2)
        return (Jp2 - Jm2) / (2j)
    
    # Rank 4 operators
    def _O40(self):
        """O_4^0 = 35J_z^4 - (30X-25)J_z^2 + 3X^2 - 6X"""
        Jz2 = self._matrix_power(self.Jz, 2)
        Jz4 = self._matrix_power(self.Jz, 4)
        I = np.eye(self.dim)
        return 35*Jz4 - (30*self.X - 25)*Jz2 + (3*self.X**2 - 6*self.X)*I
    
    def _O42(self):
        """O_4^2 = 1/4 * {J_+^2 + J_-^2, 7J_z^2 - X - 5}"""
        Jz2 = self._matrix_power(self.Jz, 2)
        Jp2 = self._matrix_power(self.Jplus, 2)
        Jm2 = self._matrix_power(self.Jminus, 2)
        A = Jp2 + Jm2
        B = 7*Jz2 - self.X*np.eye(self.dim) - 5*np.eye(self.dim)
        return self._anticommutator(A, B) / 4
    
    def _O4m2(self):
        """O_4^{-2} = 1/(4i) * {J_+^2 - J_-^2, 7J_z^2 - X - 5}"""
        Jz2 = self._matrix_power(self.Jz, 2)
        Jp2 = self._matrix_power(self.Jplus, 2)
        Jm2 = self._matrix_power(self.Jminus, 2)
        A = Jp2 - Jm2
        B = 7*Jz2 - self.X*np.eye(self.dim) - 5*np.eye(self.dim)
        return self._anticommutator(A, B) / (4j)
    
    def _O44(self):
        """O_4^4 = (J_+^4 + J_-^4)/2"""
        Jp4 = self._matrix_power(self.Jplus, 4)
        Jm4 = self._matrix_power(self.Jminus, 4)
        return (Jp4 + Jm4) / 2
    
    def _O4m4(self):
        """O_4^{-4} = (J_+^4 - J_-^4)/(2i)"""
        Jp4 = self._matrix_power(self.Jplus, 4)
        Jm4 = self._matrix_power(self.Jminus, 4)
        return (Jp4 - Jm4) / (2j)
    
    # Rank 6 operators
    def _O60(self):
        """O_6^0 = 231J_z^6 - (315X-735)J_z^4 + (105X^2-525X+294)J_z^2 - 5X^3 + 40X^2 - 60X"""
        Jz2 = self._matrix_power(self.Jz, 2)
        Jz4 = self._matrix_power(self.Jz, 4)
        Jz6 = self._matrix_power(self.Jz, 6)
        I = np.eye(self.dim)
        return (231*Jz6 - (315*self.X - 735)*Jz4 + 
                (105*self.X**2 - 525*self.X + 294)*Jz2 + 
                (-5*self.X**3 + 40*self.X**2 - 60*self.X)*I)
    
    def _O62(self):
        """O_6^2 = 1/4 * {J_+^2 + J_-^2, 33J_z^4 - (18X+123)J_z^2 + X^2 + 10X + 102}"""
        Jz2 = self._matrix_power(self.Jz, 2)
        Jz4 = self._matrix_power(self.Jz, 4)
        Jp2 = self._matrix_power(self.Jplus, 2)
        Jm2 = self._matrix_power(self.Jminus, 2)
        I = np.eye(self.dim)
        A = Jp2 + Jm2
        B = 33*Jz4 - (18*self.X + 123)*Jz2 + (self.X**2 + 10*self.X + 102)*I
        return self._anticommutator(A, B) / 4
    
    def _O6m2(self):
        """O_6^{-2} = 1/(4i) * {J_+^2 - J_-^2, 33J_z^4 - (18X+123)J_z^2 + X^2 + 10X + 102}"""
        Jz2 = self._matrix_power(self.Jz, 2)
        Jz4 = self._matrix_power(self.Jz, 4)
        Jp2 = self._matrix_power(self.Jplus, 2)
        Jm2 = self._matrix_power(self.Jminus, 2)
        I = np.eye(self.dim)
        A = Jp2 - Jm2
        B = 33*Jz4 - (18*self.X + 123)*Jz2 + (self.X**2 + 10*self.X + 102)*I
        return self._anticommutator(A, B) / (4j)
    
    def _O64(self):
        """O_6^4 = 1/4 * {J_+^4 + J_-^4, 11J_z^2 - X - 38}"""
        Jz2 = self._matrix_power(self.Jz, 2)
        Jp4 = self._matrix_power(self.Jplus, 4)
        Jm4 = self._matrix_power(self.Jminus, 4)
        I = np.eye(self.dim)
        A = Jp4 + Jm4
        B = 11*Jz2 - (self.X + 38)*I
        return self._anticommutator(A, B) / 4
    
    def _O6m4(self):
        """O_6^{-4} = 1/(4i) * {J_+^4 - J_-^4, 11J_z^2 - X - 38}"""
        Jz2 = self._matrix_power(self.Jz, 2)
        Jp4 = self._matrix_power(self.Jplus, 4)
        Jm4 = self._matrix_power(self.Jminus, 4)
        I = np.eye(self.dim)
        A = Jp4 - Jm4
        B = 11*Jz2 - (self.X + 38)*I
        return self._anticommutator(A, B) / (4j)
    
    def _O66(self):
        """O_6^6 = (J_+^6 + J_-^6)/2"""
        Jp6 = self._matrix_power(self.Jplus, 6)
        Jm6 = self._matrix_power(self.Jminus, 6)
        return (Jp6 + Jm6) / 2
    
    def _O6m6(self):
        """O_6^{-6} = (J_+^6 - J_-^6)/(2i)"""
        Jp6 = self._matrix_power(self.Jplus, 6)
        Jm6 = self._matrix_power(self.Jminus, 6)
        return (Jp6 - Jm6) / (2j)
    
    def build_hamiltonian(self, coefficients):
        """
        Build crystal field Hamiltonian from Stevens operator coefficients.
        
        Parameters:
        -----------
        coefficients : dict
            Dictionary with keys (l, m) and values B_l^m
            
        Returns:
        --------
        numpy.ndarray
            Crystal field Hamiltonian matrix
        """
        H = np.zeros((self.dim, self.dim), dtype=complex)
        
        for (l, m), B in coefficients.items():
            if B != 0:
                H += B * self.get_operator(l, m)
                
        return H
    
    def time_reversal_conjugate(self, H):
        """Apply time reversal transformation to Hamiltonian"""
        U_T = np.diag([(-1)**(self.J - m) for m in self.m_values])
        return U_T @ H.conj() @ U_T

# --- Added Schrieffer–Wolff utilities ---
def schrieffer_wolff_unitary(U, P, n_subspace, regularization=1e-12, max_angle=0.5):
    """Perform first-order Schrieffer–Wolff block diagonalization for a unitary U.
    Parameters:
        U : full unitary (N,N)
        P : projector onto target subspace (N,N) Hermitian idempotent
        n_subspace : dimension of target subspace
        regularization : small number to avoid division by zero in degeneracies
        max_angle : cap for mixing amplitude to keep expansion controlled
    Returns:
        W : dressing unitary ~ exp(S) with off-diagonal S
        P_dressed : dressed projector W P W^† (Hermitianized)
        basis_dressed : (N, n_subspace) orthonormal columns spanning dressed subspace
        U_eff : effective unitary in dressed subspace (n_subspace,n_subspace)
        info : dict with diagnostics
    Notes:
        Exact block diagonalization may be obstructed if diagonal blocks share eigenvalues with off-diagonal resonance; we regularize those terms (set mixing=0 in degenerate case).
    """
    N = U.shape[0]
    # Orthonormal basis matrices for P and Q
    evals_P, evecs_P = np.linalg.eigh(P)
    idx = np.argsort(-evals_P)[:n_subspace]
    Vp = evecs_P[:, idx]  # (N, n_subspace)
    # Q projector
    Q = np.eye(N) - P
    # Build orthonormal basis for Q via eigenvectors of Q
    evals_Q, evecs_Q = np.linalg.eigh(Q)
    idxQ = np.argsort(-evals_Q)[:N - n_subspace]
    Vq = evecs_Q[:, idxQ]
    # Blocks of U
    A = Vp.conj().T @ U @ Vp
    D = Vq.conj().T @ U @ Vq
    B = Vp.conj().T @ U @ Vq  # (n_subspace, N-n_subspace)
    # Diagonalize A and D (both unitary)
    a_vals, Ua = np.linalg.eig(A)
    d_vals, Ud = np.linalg.eig(D)
    # Transform B
    B_tilde = Ua.conj().T @ B @ Ud
    # Solve (A S - S D) = -B  => (a_i - d_j) s_ij = -b_ij
    S_tilde = np.zeros_like(B_tilde, dtype=complex)
    resonant = 0
    for i in range(a_vals.shape[0]):
        for j in range(d_vals.shape[0]):
            denom = a_vals[i] - d_vals[j]
            if abs(denom) < regularization:
                # cannot remove resonant coupling exactly; leave zero (S_ij=0)
                resonant += 1
            else:
                S_tilde[i, j] = -B_tilde[i, j] / denom
    # Transform back
    S_PQ = Ua @ S_tilde @ Ud.conj().T
    # Optional cap to keep perturbative
    norm_S = np.linalg.norm(S_PQ, ord=2)
    if norm_S > max_angle and norm_S > 0:
        S_PQ *= max_angle / norm_S
    # Assemble S (anti-Hermitian off-diagonal)
    S = np.zeros((N, N), dtype=complex)
    # P->Q block: Vp S_PQ Vq^† ; Q->P block: - ( ... )^†
    S_PQ_full = Vp @ S_PQ @ Vq.conj().T
    S += S_PQ_full - S_PQ_full.conj().T  # ensures anti-Hermitian
    # Approximate W ≈ I + S + 1/2 S^2 (sufficient for small S)
    S2 = S @ S
    W = np.eye(N, dtype=complex) + S + 0.5 * S2
    # (Optionally re-orthonormalize via QR to ensure unitary to this order)
    # Use polar decomposition to unitarize W
    X = W.conj().T @ W
    evalsX, evecsX = np.linalg.eigh(X)
    X_inv_sqrt = evecsX @ np.diag(1/np.sqrt(evalsX)) @ evecsX.conj().T
    W = W @ X_inv_sqrt
    # Dressed projector
    P_dressed = W @ P @ W.conj().T
    # Hermitianize & idempotent cleanup
    P_dressed = 0.5 * (P_dressed + P_dressed.conj().T)
    # Extract dressed basis
    evalsPd, vecsPd = np.linalg.eigh(P_dressed)
    order = np.argsort(-evalsPd)[:n_subspace]
    basis_dressed = vecsPd[:, order]
    # Effective unitary inside dressed subspace
    U_eff = basis_dressed.conj().T @ U @ basis_dressed
    # Unitarize inside subspace via polar in case of residual error
    Y = U_eff.conj().T @ U_eff
    evalsY, vecsY = np.linalg.eigh(Y)
    Y_inv_sqrt = vecsY @ np.diag(1/np.sqrt(evalsY)) @ vecsY.conj().T
    U_eff = U_eff @ Y_inv_sqrt
    info = {
        'norm_S': norm_S,
        'resonant_couplings': resonant,
        'unitarity_error_original': np.linalg.norm((Vp.conj().T @ U @ Vp).conj().T @ (Vp.conj().T @ U @ Vp) - np.eye(n_subspace)),
        'unitarity_error_eff': np.linalg.norm(U_eff.conj().T @ U_eff - np.eye(n_subspace))
    }
    return W, P_dressed, basis_dressed, U_eff, info

# Example usage
if __name__ == "__main__":
    # Create Stevens operators for J=6
    stevens = StevensOperators(J=6)
    
    # Define crystal field parameters
    coefficients = {
        (2, 0): -5.29e-1,
        (2, 2): -1.35e-1,
        (2, -2): 12.79e-1,
        (4, 0): -0.13e-3,
        (4, 2): -1.7e-3,
        (4, -2): 3.29e-3,
        (4, 4): -1.22e-3,
        (4, -4): -9.57e-3,
        (6, 0): 0.2e-5,
        (6, 2): -1.1e-5,
        (6, -2): -0.9e-5,
        (6, 4): 6.1e-5,
        (6, -4): 0.3e-5,
        (6, 6): -0.9e-5,
        (6, -6): 0
    }
    
    # Build Hamiltonian
    H = stevens.build_hamiltonian(coefficients)
    
    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues -= eigenvalues.min()
    print("Eigenvalues are: ")
    print(eigenvalues)
    # Apply time reversal symmetry to eigenvectors
    # Time reversal operator for J states: T|J,m> = (-1)^(J-m)|J,-m>
    T_phases = np.array([(-1)**(stevens.J - m) for m in stevens.m_values])
    
    # Make eigenvectors time reversal symmetric by choosing appropriate global phase
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        
        # Apply time reversal: T|psi> = T_phases * conj(vec[::-1])
        vec_T = T_phases * np.conj(vec[::-1])
        
        # Find phase that makes vec = vec_T (up to a global phase)
        # We can use the component with largest magnitude for stability
        max_idx = np.argmax(np.abs(vec))
        if np.abs(vec[max_idx]) > 1e-10:
            # Calculate the phase difference
            phase = vec_T[max_idx] / vec[max_idx]
            phase = phase / np.abs(phase)  # Normalize to unit magnitude
            
            # Apply the phase rotation
            eigenvectors[:, i] = vec * np.sqrt(phase)
    
    # Subspace of the 3 lowest states
    subspace_basis = eigenvectors[:, :3]

    # Print the subspace basis
    print("Subspace basis for the 3 lowest states:")
    print(subspace_basis.T)

    # Print the subspace basis in a clean LaTeX format
    print("\nSubspace basis in LaTeX format:")
    m_values = stevens.m_values
    for i in range(subspace_basis.shape[1]):
        vec = subspace_basis[:, i]
        state_str = f"|\\psi_{i}\\rangle = "
        first_term = True
        for j, coeff in enumerate(vec):
            if abs(coeff) > 1e-4:
                m = m_values[j]
                
                # Format coefficient
                if abs(coeff.imag) < 1e-4: # Real
                    val = coeff.real
                    sign = "+" if val > 0 else "-"
                    if first_term and val > 0:
                        sign = ""
                    coeff_str = f"{sign} {abs(val):.4f}"
                else: # Complex
                    sign = "+" if coeff.real >= 0 else "-"
                    if first_term and coeff.real >= 0:
                        sign = ""
                    
                    # Handle pure imaginary
                    if abs(coeff.real) < 1e-4:
                         sign_imag = "+" if coeff.imag > 0 else "-"
                         coeff_str = f"{sign_imag} {abs(coeff.imag):.4f}i"
                         if first_term and coeff.imag > 0:
                             coeff_str = f"{abs(coeff.imag):.4f}i"
                    else:
                        sign_imag = "+" if coeff.imag > 0 else "-"
                        coeff_str = f"{sign} ({abs(coeff.real):.4f} {sign_imag} {abs(coeff.imag):.4f}i)"

                if first_term:
                    state_str += f"{coeff_str.strip()} |{m:+.0f}\\rangle "
                    first_term = False
                else:
                    state_str += f"{coeff_str} |{m:+.0f}\\rangle "
        print(state_str)
    
    # Dot products (overlaps) between the lowest 3 eigenvectors
    overlaps = eigenvectors[:, :3].conj().T @ eigenvectors[:, :3]
    print("\nDot products (overlap matrix) between the lowest 3 eigenvectors:")
    np.set_printoptions(precision=6, suppress=True)
    print(overlaps)
    np.set_printoptions()

    # Jx and Jy operators
    Jx = (stevens.Jplus + stevens.Jminus) / 2
    Jy = (stevens.Jplus - stevens.Jminus) / (2j)
    Jz = stevens.Jz


    


    # Project operators into the subspace
    Jx_proj = subspace_basis.T.conj() @ Jx @ subspace_basis
    Jy_proj = subspace_basis.T.conj() @ Jy @ subspace_basis
    Jz_proj = subspace_basis.T.conj() @ Jz @ subspace_basis

    # Set a threshold for small numbers
    threshold = 1e-9

    # Clean up matrices for printing
    Jx_proj[np.abs(Jx_proj) < threshold] = 0
    Jy_proj[np.abs(Jy_proj) < threshold] = 0
    Jz_proj[np.abs(Jz_proj) < threshold] = 0

    # Set numpy print options for cleaner output
    np.set_printoptions(precision=4, suppress=True)

    print("\nProjected Jx matrix in the subspace of the 3 lowest states:")
    print(Jx_proj)
    
    print("\nProjected Jy matrix in the subspace of the 3 lowest states:")
    print(Jy_proj)
    
    print("\nProjected Jz matrix in the subspace of the 3 lowest states:")
    print(Jz_proj)

    subspace_eig = np.diag(eigenvalues[:3])

    print("\nEigenvalues in the subspace of the 3 lowest states:")
    print(subspace_eig)
    # Admixture of higher states into the low-energy subspace
    JXX = Jx_proj + subspace_eig
    print("\nAdmixture of higher states into the low-energy subspace (Jx_proj + diag(E)):")
    print(JXX)
    print("With eigenvalues and eigenvectors:")
    evals, evecs = np.linalg.eigh(JXX)
    print("Eigenvalues:")
    print(evals)
    print("Eigenvectors:")
    print(evecs)


    # Time reversal operator
    # T|J,m> = (-1)^(J-m)|J,-m>
    # T is anti-unitary, T = U_T K where K is complex conjugation
    # U_T |J,m> = (-1)^(J-m)|J,-m>
    dim = stevens.dim
    U_T = np.zeros((dim, dim))
    m_values = stevens.m_values
    for i in range(dim):
        m = m_values[i]
        phase = (-1)**(stevens.J - m)
        # Find index for -m
        j = np.where(m_values == -m)[0][0]
        U_T[j, i] = phase

    # Project the unitary part of the time reversal operator
    UT_proj = subspace_basis.T.conj() @ U_T @ subspace_basis.conj()

    # Clean up matrix for printing
    UT_proj[np.abs(UT_proj) < threshold] = 0

    print("\nProjected Time Reversal operator (unitary part) in the subspace:")
    print("Note: T = U_T K, where K is complex conjugation.")
    print("The matrix below is the projection of U_T.")
    print(UT_proj)

    # R_n(pi) = exp(-i * pi * J_n)
    # Rz is simple because Jz is diagonal in |J,m> basis
    Rz_pi_full = np.diag(np.exp(-1j * np.pi * m_values))

    # For Jx use spectral decomposition (Hermitian)
    evals_x, evecs_x = np.linalg.eigh(Jx)
    Rx_pi_full = evecs_x @ np.diag(np.exp(-1j * np.pi * evals_x)) @ evecs_x.conj().T

    # Project into the low-energy 3D subspace
    Rz_pi_proj = subspace_basis.T.conj() @ Rz_pi_full @ subspace_basis
    Rx_pi_proj = subspace_basis.T.conj() @ Rx_pi_full @ subspace_basis

    # Clean small numerical noise
    Rz_pi_proj[np.abs(Rz_pi_proj) < threshold] = 0
    Rx_pi_proj[np.abs(Rx_pi_proj) < threshold] = 0

    print("\nProjected pi rotation about z (Rz(pi)) in the subspace:")
    print(Rz_pi_proj)
    print("\nProjected pi rotation about x (Rx(pi)) in the subspace:")
    print(Rx_pi_proj)

    # Unitarity checks in full space
    err_Rz_full = np.linalg.norm(Rz_pi_full.conj().T @ Rz_pi_full - np.eye(dim))
    err_Rx_full = np.linalg.norm(Rx_pi_full.conj().T @ Rx_pi_full - np.eye(dim))

    # Unitarity checks in projected subspace
    I3 = np.eye(subspace_basis.shape[1])
    err_Rz_proj = np.linalg.norm(Rz_pi_proj.conj().T @ Rz_pi_proj - I3)
    err_Rx_proj = np.linalg.norm(Rx_pi_proj.conj().T @ Rx_pi_proj - I3)

    print("\nUnitarity check (Frobenius norms of U†U - I):")
    print(f"Full Rz(pi) error: {err_Rz_full:.2e}")
    print(f"Full Rx(pi) error: {err_Rx_full:.2e}")
    print(f"Proj Rz(pi) error: {err_Rz_proj:.2e}")
    print(f"Proj Rx(pi) error: {err_Rx_proj:.2e}")

    # Reset print options to default if needed elsewhere
    np.set_printoptions()


