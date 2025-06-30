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
    
    print("Energy levels (relative to ground state):")
    print(eigenvalues)

    print("\nEigenvectors (time-reversal symmetric):")
    for i in range(eigenvectors.shape[1]):
        print(f"State {i}: {eigenvectors[:, i]}")
    ground_state = eigenvectors[:, 0]
    print("\nGround state:")
    print(ground_state)
    
    # Verify time reversal symmetry
    print("\nTime reversal symmetry check for ground state:")
    gs_T = T_phases * np.conj(ground_state[::-1])
    print(f"Max difference: {np.max(np.abs(ground_state - gs_T))}")
    
    # Test individual operator
    O2n2 = stevens.get_operator(2, -2)
    O22 = stevens.get_operator(2, 2)
    print(f"\nTime reversal check for O_2^{{-2}}:")
    print(f"Equal? {np.allclose(stevens.time_reversal_conjugate(O2n2), O2n2)}")