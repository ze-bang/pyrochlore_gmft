import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the h vectors
h01 = np.array([0, 1/4, -1/4])
h02 = np.array([-1/4, 0, 1/4])
h03 = np.array([1/4, -1/4, 0])
h10 = -h01  # h10 = -h01 by antisymmetry
h20 = -h02  # h20 = -h02
h30 = -h03  # h30 = -h03
h12 = np.array([1/4, 1/4, 0])
h21 = -h12  # h21 = -h12
h13 = np.array([-1/4, 0, -1/4])
h31 = -h13  # h31 = -h13
h23 = np.array([0, 1/4, 1/4])
h32 = -h23  # h32 = -h23

def construct_Z_matrix(k, K0=1.0, K1=1.0, K2=1.0, K3=1.0):
    """
    Construct the Z matrix for a given k-point.
    
    Parameters:
    -----------
    k : array-like, shape (3,)
        The k-vector in reciprocal space
    K0, K1, K2, K3 : float
        The coupling constants (default all 1.0)
    
    Returns:
    --------
    Z : ndarray, shape (4, 4)
        The Z matrix (pure imaginary antisymmetric)
    """
    k = np.array(k)
    
    # Calculate sin(k·h) for all pairs
    Z = np.zeros((4, 4), dtype=complex)
    
    # Row 0
    Z[0, 1] = np.sqrt(K1) * np.sin(np.dot(k, h01))
    Z[0, 2] = np.sqrt(K2) * np.sin(np.dot(k, h02))
    Z[0, 3] = np.sqrt(K3) * np.sin(np.dot(k, h03))
    
    # Row 1
    Z[1, 0] = np.sqrt(K0) * np.sin(np.dot(k, h10))
    Z[1, 2] = np.sqrt(K2) * np.sin(np.dot(k, h12))
    Z[1, 3] = np.sqrt(K3) * np.sin(np.dot(k, h13))
    
    # Row 2
    Z[2, 0] = np.sqrt(K0) * np.sin(np.dot(k, h20))
    Z[2, 1] = np.sqrt(K1) * np.sin(np.dot(k, h21))
    Z[2, 3] = np.sqrt(K3) * np.sin(np.dot(k, h23))
    
    # Row 3
    Z[3, 0] = np.sqrt(K0) * np.sin(np.dot(k, h30))
    Z[3, 1] = np.sqrt(K1) * np.sin(np.dot(k, h31))
    Z[3, 2] = np.sqrt(K2) * np.sin(np.dot(k, h32))
    
    # Multiply by -2i
    Z = -2j * Z
    
    return Z

def compute_K_values(Jpm, Jzz, B, field_dir):
    """
    Compute K0, K1, K2, K3 coupling constants based on the g0 formula.
    
    Parameters:
    -----------
    Jpm : float
        Transverse exchange coupling
    Jzz : float
        Ising exchange coupling
    B : float
        Magnetic field strength
    field_dir : array-like, shape (3,)
        Magnetic field direction (will be normalized)
    
    Returns:
    --------
    K_values : ndarray, shape (4,)
        [K0, K1, K2, K3] coupling constants for each sublattice
    """
    # Define sublattice-dependent local z-axes (same as in QuantumSpinIcePhoton)
    z0 = 1/np.sqrt(3)*np.array([1, 1, 1])
    z1 = 1/np.sqrt(3)*np.array([1, -1, -1])
    z2 = 1/np.sqrt(3)*np.array([-1, 1, -1])
    z3 = 1/np.sqrt(3)*np.array([-1, -1, 1])
    z_local_gcc_vec = np.array([z0, z1, z2, z3])
    
    # Normalize field direction
    field_dir = np.array(field_dir)
    field_dir = field_dir / np.linalg.norm(field_dir) if np.linalg.norm(field_dir) > 0 else np.array([0, 0, 1])
    
    # Compute g0 for each sublattice
    # g0 = 24 * |Jpm|^3 / Jzz^2 + 10 * Jpm^2 / |Jzz|^3 * B^2 * (z_local · field_dir)^2
    base_term = 12 * np.abs(Jpm**3) / Jzz**2
    
    # Compute field-dependent term for each sublattice
    z_dot_field = np.dot(z_local_gcc_vec, field_dir)  # Shape: (4,)
    field_term = 5 * Jpm**2 / np.abs(Jzz**3) * B**2 * z_dot_field**2
    
    # Total K values for each sublattice
    K_values = base_term + field_term
    
    return K_values

def diagonalize_Z(k, K0=1.0, K1=1.0, K2=1.0, K3=1.0, return_all=False):
    """
    Diagonalize the Z matrix at a given k-point.
    
    Parameters:
    -----------
    k : array-like
        The k-vector
    K0, K1, K2, K3 : float
        Coupling constants
    return_all : bool
        If True, return eigenvalues, eigenvectors, and the matrix itself
    
    Returns:
    --------
    eigenvalues : ndarray
        The eigenvalues (sorted by real part)
    eigenvectors : ndarray (if return_all=True)
        The eigenvectors (columns)
    Z : ndarray (if return_all=True)
        The Z matrix
    """
    Z = construct_Z_matrix(k, K0, K1, K2, K3)
    
    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eig(Z)
    
    # Sort by real part (since they come in ±pairs for antisymmetric matrices)
    idx = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if return_all:
        return eigenvalues, eigenvectors, Z
    else:
        return eigenvalues

# ============================================================================
# MAIN CALCULATION: [111] MAGNETIC FIELD
# ============================================================================
print("=" * 70)
print("BAND STRUCTURE WITH [111] MAGNETIC FIELD")
print("=" * 70)

# Physical parameters
Jpm = 0.1
Jzz = 1.0
B = 1
field_dir = np.array([1, 1, 1])  # [111] direction

# Compute K values for [111] field
K_vals = compute_K_values(Jpm, Jzz, B, field_dir)
print(f"\nPhysical parameters:")
print(f"  Jpm = {Jpm}")
print(f"  Jzz = {Jzz}")
print(f"  B = {B}")
print(f"  Field direction = [111]")

print(f"\nComputed K values:")
print(f"  K0 = {K_vals[0]:.6f}")
print(f"  K1 = {K_vals[1]:.6f}")
print(f"  K2 = {K_vals[2]:.6f}")
print(f"  K3 = {K_vals[3]:.6f}")

# Calculate band structure along high-symmetry path
# Path: Γ → X → W → K → Γ → L → U → W → L → K
def get_high_symmetry_points():
    """Get high-symmetry points for FCC lattice in units of 2π/a"""
    return {
        'Γ': np.array([0, 0, 0]),
        'X': np.array([0.5, 0, 0.5]),
        'W': np.array([0.5, 0.25, 0.75]),
        'K': np.array([0.375, 0.375, 0.75]),
        'L': np.array([0.5, 0.5, 0.5]),
        'U': np.array([0.625, 0.25, 0.625])
    }

def generate_kpath(path_labels, npoints=100):
    """Generate k-points along a path"""
    hs_points = get_high_symmetry_points()
    kpath = []
    k_distance = []
    total_dist = 0
    
    for i in range(len(path_labels) - 1):
        k1 = hs_points[path_labels[i]]
        k2 = hs_points[path_labels[i+1]]
        
        for j in range(npoints):
            t = j / npoints
            k = k1 + t * (k2 - k1)
            kpath.append(k * 2 * np.pi)
            
            if len(kpath) > 1:
                total_dist += np.linalg.norm(kpath[-1] - kpath[-2])
            k_distance.append(total_dist)
    
    # Add final point
    kpath.append(hs_points[path_labels[-1]] * 2 * np.pi)
    if len(kpath) > 1:
        total_dist += np.linalg.norm(kpath[-1] - kpath[-2])
    k_distance.append(total_dist)
    
    return np.array(kpath), np.array(k_distance)

# Generate k-path
path = ['Γ', 'X', 'W', 'K', 'Γ', 'L', 'U', 'W', 'L', 'K']
kpath, kdist = generate_kpath(path, npoints=50)

print(f"\nCalculating band structure along: {' → '.join(path)}")
print(f"Total k-points: {len(kpath)}")

# Calculate eigenvalues along the path with [111] field K values
bands = []
for k in kpath:
    eigenvals = diagonalize_Z(k, K0=K_vals[0], K1=K_vals[1], K2=K_vals[2], K3=K_vals[3])
    # Take imaginary part (eigenvalues are pure imaginary for antisymmetric matrices)
    bands.append(eigenvals.real)

bands = np.array(bands)

print(f"\nBand structure shape: {bands.shape}")
print(f"Energy range: [{bands.min():.4f}, {bands.max():.4f}]")

# Plot band structure
plt.figure(figsize=(10, 6))
for i in range(bands.shape[1]):
    plt.plot(kdist, bands[:, i], 'b-', linewidth=1.5)

# Add vertical lines at high-symmetry points
hs_positions = [0]
for i in range(len(path) - 1):
    idx = (i + 1) * 50
    if idx < len(kdist):
        hs_positions.append(kdist[idx])

for pos in hs_positions:
    plt.axvline(pos, color='k', linestyle='--', alpha=0.3)

plt.xticks(hs_positions, path)
plt.ylabel('Energy (imaginary part of eigenvalues)')
plt.xlabel('k-path')
plt.title('Band Structure of Z(k) Matrix')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Z_matrix_band_structure.png', dpi=300)
print("\nBand structure plot saved as 'Z_matrix_band_structure.png'")

plt.show()

