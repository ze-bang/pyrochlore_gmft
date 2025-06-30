import numpy as np
from stevens_operator import *

# Since the Hamiltonian is time-reversal invariant, let's check why the eigenstates
# are not showing time-reversal even behavior

print("Investigating the eigenstate time-reversal properties:")
print("="*60)

# Time reversal operator
def time_reversal_matrix():
    T_matrix = np.zeros((13, 13), dtype=complex)
    for i, m in enumerate(range(-6, 7)):  # m goes from -6 to 6
        j = 12 - i  # index of |-m> state
        phase = (-1)**(6 - m)
        T_matrix[j, i] = phase
    return T_matrix

T_op = time_reversal_matrix()

# Get eigenvalues and eigenvectors
E, V = np.linalg.eigh(H)
print(f"Eigenvalues: {E}")
print(f"Energy differences: {np.diff(E)}")

# Check for degeneracies (Kramers pairs)
degeneracy_threshold = 1e-10
for i in range(len(E)):
    for j in range(i+1, len(E)):
        if abs(E[i] - E[j]) < degeneracy_threshold:
            print(f"Degenerate states found: E[{i}] ≈ E[{j}] = {E[i]:.6f}")

print("\nAnalyzing individual eigenstates:")
for i in range(min(6, len(E))):  # Check first 6 eigenstates
    psi = V[:, i]
    
    # Apply time reversal: T|psi> = T_op * conj(psi)
    psi_time_reversed = T_op @ np.conj(psi)
    
    # Check if it's still an eigenstate with the same energy
    H_psi_tr = H @ psi_time_reversed
    expected = E[i] * psi_time_reversed
    is_eigenstate = np.allclose(H_psi_tr, expected, atol=1e-10)
    
    # Check overlap with original state
    overlap_with_self = np.abs(np.vdot(psi, psi_time_reversed))**2
    
    print(f"\nEigenstate {i} (E = {E[i]:.6f}):")
    print(f"  |psi>: {psi}")
    print(f"  T|psi*>: {psi_time_reversed}")
    print(f"  Is T|psi*> an eigenstate? {is_eigenstate}")
    print(f"  Overlap |<psi|T|psi*>|^2: {overlap_with_self:.6f}")
    
    # Check if T|psi*> is proportional to any other eigenstate
    found_partner = False
    for j in range(len(E)):
        if i != j:
            other_psi = V[:, j]
            overlap = np.abs(np.vdot(other_psi, psi_time_reversed))**2
            if overlap > 0.9:  # High overlap
                print(f"  T|psi*> ≈ |psi_{j}> with overlap {overlap:.6f}")
                print(f"  Energy difference: {abs(E[j] - E[i]):.2e}")
                found_partner = True
                break
    
    if not found_partner and overlap_with_self < 0.9:
        print(f"  Warning: T|psi*> is not well-represented in the eigenspace!")

# Let's also check if the issue is in the Kramers theorem expectation
print(f"\n" + "="*60)
print("Kramers theorem analysis:")
print("For integer J (J=6), Kramers theorem doesn't apply.")
print("States can be time-reversal even without degeneracy.")

# Check the specific structure of the ground state
print(f"\nGround state analysis:")
psi0 = V[:, 0]
print(f"Ground state coefficients: {psi0}")
print(f"Ground state |coefficients|: {np.abs(psi0)}")

# Check if it has the expected symmetry structure
# For time-reversal even state: c_m = c_{-m}^* (up to phase)
print(f"\nChecking symmetry of ground state coefficients:")
for m in range(7):  # m from 0 to 6
    if m == 0:
        print(f"m=0: c_0 = {psi0[6]:.6f} (should be real for time-reversal even)")
    else:
        c_plus_m = psi0[6 + m]
        c_minus_m = psi0[6 - m]
        print(f"m=±{m}: c_{m} = {c_plus_m:.6f}, c_{-m} = {c_minus_m:.6f}")
        print(f"        |c_{m}| = {abs(c_plus_m):.6f}, |c_{-m}| = {abs(c_minus_m):.6f}")
        print(f"        Are |c_m| = |c_{-m}|? {np.allclose(abs(c_plus_m), abs(c_minus_m), atol=1e-6)}")

# The eigenstates should be real for a time-reversal invariant Hamiltonian
print(f"\nAre the eigenstates real?")
for i in range(min(5, len(E))):
    psi = V[:, i]
    is_real = np.allclose(psi.imag, 0, atol=1e-10)
    print(f"Eigenstate {i}: real = {is_real}, max_imag = {np.max(np.abs(psi.imag)):.2e}")
