"""Core GMFT library for pyrochlore lattice spin-liquid calculations.

This package contains the foundational modules for gauge mean-field theory:

- ``flux_stuff``     : Gauge flux computation on pyrochlore plaquettes
- ``misc_helper``    : Lattice constants, BZ generation, coordinate transforms
- ``pyrochlore_gmft``: Main GMFT solver (Hamiltonian, self-consistency, piFluxSolver)
- ``pyrochlore_exclusive_boson``: Exclusive-boson variant of the Hamiltonian
- ``observables``    : Physical observables (SSSF, DSSF, structure factors)
- ``phase_diagram``  : Phase diagram computation and scanning
- ``variation_flux`` : Variational flux optimisation
- ``monte_carlo``    : Classical Monte Carlo simulation (MPI-parallelised)
"""
