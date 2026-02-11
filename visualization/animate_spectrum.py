#!/usr/bin/env python3
"""
Standalone script to animate integrated spectrum as a function of magnetic field strength.
This can be run independently to create animations from existing data files.

Usage:
    python animate_spectrum.py [--dir DIRECTORY] [--output OUTPUT_FILE] [--fps FPS]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import argparse

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["CMU Serif"]
plt.rcParams["font.size"] = 14


def animate_integrated_spectrum(dir='DSSF_CZO', H_min=0, H_max=4, n_H=20, 
                                output_filename='integrated_spectrum_animation.gif', 
                                fps=5, dpi=100):
    """
    Create an animation of integrated spectrum as a function of magnetic field strength.
    
    Parameters:
    -----------
    dir : str
        Directory containing the integrated spectrum files
    H_min : float
        Minimum magnetic field value (in Tesla)
    H_max : float
        Maximum magnetic field value (in Tesla)
    n_H : int
        Number of field points
    output_filename : str
        Name of the output animation file
    fps : int
        Frames per second for the animation
    dpi : int
        Resolution of the output animation
    """
    H_values = np.linspace(H_min, H_max, n_H)
    mu_B = 0.05788  # meV/T
    g = 2.24
    
    # Load all spectrum data
    spectra = []
    omega_vals = None
    valid_H_values = []
    valid_B_values = []
    
    for h in H_values:
        B = h * mu_B / g
        filename = f"{dir}/integrated_spectrum_{B}.txt"
        if os.path.exists(filename):
            try:
                data = np.loadtxt(filename)
                if omega_vals is None:
                    omega_vals = data[:, 0]
                spectra.append(data[:, 1])
                valid_H_values.append(h)
                valid_B_values.append(B)
            except Exception as e:
                print(f"Warning: Error loading {filename}: {e}")
        else:
            print(f"Warning: File {filename} not found, skipping H = {h:.3f} T")
    
    if len(spectra) == 0:
        print(f"No spectrum files found in {dir}!")
        print(f"Looking for files like: integrated_spectrum_*.txt")
        return None
    
    print(f"Found {len(spectra)} spectrum files")
    spectra = np.array(spectra)
    valid_H_values = np.array(valid_H_values)
    valid_B_values = np.array(valid_B_values)
    
    # Find global min/max for consistent y-axis
    y_min = np.min(spectra)
    y_max = np.max(spectra)
    y_range = y_max - y_min
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', linewidth=2.5)
    ax.set_xlim(omega_vals[0], omega_vals[-1])
    ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
    ax.set_xlabel(r'Energy (meV)', fontsize=16)
    ax.set_ylabel(r'Intensity (arb. units)', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Text annotation for field strength
    field_text = ax.text(0.98, 0.95, '', transform=ax.transAxes, 
                        fontsize=14, verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Title
    ax.set_title(r'Integrated Dynamical Structure Factor vs Magnetic Field', 
                fontsize=18, pad=20)
    
    # Make the plot look nicer
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, labelsize=13)
    
    # Initialization function
    def init():
        line.set_data([], [])
        field_text.set_text('')
        return line, field_text
    
    # Animation function
    def animate(i):
        line.set_data(omega_vals, spectra[i])
        field_text.set_text(f'H = {valid_H_values[i]:.3f} T\nB = {valid_B_values[i]:.5f} meV')
        return line, field_text
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(valid_H_values), interval=1000//fps, 
                        blit=True, repeat=True)
    
    # Save animation
    print(f"Saving animation to {output_filename}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_filename, writer=writer, dpi=dpi)
    print(f"Animation saved successfully to {output_filename}")
    print(f"  - Total frames: {len(valid_H_values)}")
    print(f"  - Duration: {len(valid_H_values)/fps:.1f} seconds")
    print(f"  - Field range: {valid_H_values[0]:.3f} - {valid_H_values[-1]:.3f} T")
    
    plt.close()
    
    return anim


def main():
    parser = argparse.ArgumentParser(
        description='Create animation of integrated spectrum vs magnetic field'
    )
    parser.add_argument('--dir', type=str, default='DSSF_CZO',
                       help='Directory containing integrated spectrum files (default: DSSF_CZO)')
    parser.add_argument('--output', type=str, default='integrated_spectrum_animation.gif',
                       help='Output filename for animation (default: integrated_spectrum_animation.gif)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second (default: 5)')
    parser.add_argument('--dpi', type=int, default=100,
                       help='Resolution DPI (default: 100)')
    parser.add_argument('--H-min', type=float, default=0,
                       help='Minimum magnetic field in Tesla (default: 0)')
    parser.add_argument('--H-max', type=float, default=4,
                       help='Maximum magnetic field in Tesla (default: 4)')
    parser.add_argument('--n-H', type=int, default=20,
                       help='Number of field points (default: 20)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory {args.dir} does not exist!")
        return
    
    animate_integrated_spectrum(
        dir=args.dir,
        H_min=args.H_min,
        H_max=args.H_max,
        n_H=args.n_H,
        output_filename=args.output,
        fps=args.fps,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
