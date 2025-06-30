import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse
import sys

import matplotlib.pyplot as plt

def read_data(filename):
    """Read energy and intensity data from a two-column text file."""
    try:
        data = np.loadtxt(filename)
        if data.shape[1] != 2:
            raise ValueError("File must contain exactly two columns")
        energy = data[:, 0]
        intensity = data[:, 1]
        return energy, intensity
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def gaussian_resolution(energy, intensity, resolution_fwhm):
    """
    Convolve intensity with Gaussian experimental resolution.
    
    Parameters:
    - energy: array of energy values
    - intensity: array of intensity values
    - resolution_fwhm: Full Width at Half Maximum of the Gaussian resolution function
    """
    # Calculate energy spacing
    energy_spacing = np.mean(np.diff(energy))
    
    # Convert FWHM to standard deviation
    sigma = resolution_fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Convert sigma from energy units to array index units
    sigma_indices = sigma / energy_spacing
    
    # Apply Gaussian convolution
    convolved_intensity = gaussian_filter1d(intensity, sigma_indices)
    
    return convolved_intensity

def lorentzian_resolution(energy, intensity, resolution_fwhm):
    """
    Convolve intensity with Lorentzian experimental resolution.
    
    Parameters:
    - energy: array of energy values
    - intensity: array of intensity values
    - resolution_fwhm: Full Width at Half Maximum of the Lorentzian resolution function
    """
    # Create Lorentzian kernel
    energy_spacing = np.mean(np.diff(energy))
    kernel_size = int(5 * resolution_fwhm / energy_spacing)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel_energy = np.linspace(-kernel_size//2 * energy_spacing, 
                                kernel_size//2 * energy_spacing, 
                                kernel_size)
    
    gamma = resolution_fwhm / 2
    lorentzian = (gamma / np.pi) / (kernel_energy**2 + gamma**2)
    lorentzian = lorentzian / np.sum(lorentzian)  # Normalize
    
    # Apply convolution
    convolved_intensity = np.convolve(intensity, lorentzian, mode='same')
    
    return convolved_intensity

def save_data(filename, energy, intensity):
    """Save convolved data to a new file."""
    output_data = np.column_stack((energy, intensity))
    output_filename = filename.replace('.txt', '_convolved.txt')
    np.savetxt(output_filename, output_data, fmt='%.6e', 
               header='Energy\tIntensity (convolved)', delimiter='\t')
    print(f"Convolved data saved to: {output_filename}")
    return output_filename

def plot_comparison(energy, original_intensity, convolved_intensity, resolution_type, resolution_fwhm):
    """Plot original and convolved data for comparison."""
    plt.figure(figsize=(10, 6))
    plt.plot(energy, original_intensity, 'b-', label='Original', alpha=0.7)
    plt.plot(energy, convolved_intensity, 'r-', label=f'Convolved ({resolution_type}, FWHM={resolution_fwhm})', linewidth=2)
    plt.xlabel('Energy')
    plt.ylabel('Intensity')
    plt.title('Experimental Resolution Convolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Convolve spectral data with experimental resolution')
    parser.add_argument('filename', help='Input text file with two columns (energy, intensity)')
    parser.add_argument('--resolution', type=float, default=0.1, 
                        help='Resolution FWHM in energy units (default: 0.1)')
    parser.add_argument('--type', choices=['gaussian', 'lorentzian'], default='gaussian',
                        help='Type of resolution function (default: gaussian)')
    parser.add_argument('--plot', action='store_true', help='Show comparison plot')
    parser.add_argument('--no-save', action='store_true', help='Do not save output file')
    
    args = parser.parse_args()
    
    # Read data
    energy, intensity = read_data(args.filename)
    
    # Apply convolution
    if args.type == 'gaussian':
        convolved_intensity = gaussian_resolution(energy, intensity, args.resolution)
    else:
        convolved_intensity = lorentzian_resolution(energy, intensity, args.resolution)
    
    # Save results
    if not args.no_save:
        save_data(args.filename, energy, convolved_intensity)
    
    # Plot if requested
    if args.plot:
        plot_comparison(energy, intensity, convolved_intensity, args.type, args.resolution)
    
    # Print summary
    print(f"\nConvolution completed:")
    print(f"  Resolution type: {args.type}")
    print(f"  Resolution FWHM: {args.resolution}")
    print(f"  Energy range: {energy.min():.3f} to {energy.max():.3f}")

if __name__ == "__main__":
    main()