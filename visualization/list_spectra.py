#!/usr/bin/env python3
"""
List available spectrum files and preview field values.
Useful to check what data is available before creating animations.
"""

import os
import numpy as np
import glob

def list_spectrum_files(directory='DSSF_CZO'):
    """List all integrated spectrum files in a directory and extract field values."""
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return
    
    # Find all integrated spectrum files
    pattern = os.path.join(directory, 'integrated_spectrum_*.txt')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No integrated spectrum files found in {directory}")
        return
    
    print(f"\n{'='*70}")
    print(f"Spectrum Files in: {directory}")
    print(f"{'='*70}\n")
    
    mu_B = 0.05788  # meV/T
    g = 2.24
    
    B_values = []
    H_values = []
    
    for i, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        # Extract B value from filename
        B_str = filename.replace('integrated_spectrum_', '').replace('.txt', '')
        try:
            B = float(B_str)
            H = B * g / mu_B
            B_values.append(B)
            H_values.append(H)
            
            # Check file size
            file_size = os.path.getsize(filepath)
            
            print(f"{i:3d}. {filename}")
            print(f"     H = {H:8.4f} T,  B = {B:10.6f} meV,  Size = {file_size:7d} bytes")
            
        except ValueError:
            print(f"{i:3d}. {filename} - Could not parse B value")
    
    if B_values:
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"{'='*70}")
        print(f"Total files found: {len(files)}")
        print(f"H field range:     {min(H_values):.4f} - {max(H_values):.4f} T")
        print(f"B field range:     {min(B_values):.6f} - {max(B_values):.6f} meV")
        
        # Calculate spacing
        if len(H_values) > 1:
            H_spacing = np.diff(sorted(H_values))
            print(f"H spacing:         {np.mean(H_spacing):.4f} T (avg), {np.std(H_spacing):.4f} T (std)")
        print()
        
        # Preview first file
        first_file = files[0]
        try:
            data = np.loadtxt(first_file)
            print(f"Data format preview from {os.path.basename(first_file)}:")
            print(f"  Energy range:  {data[0,0]:.4f} - {data[-1,0]:.4f} meV")
            print(f"  Data points:   {len(data)}")
            print(f"  Max intensity: {np.max(data[:,1]):.4e}")
            print()
        except Exception as e:
            print(f"Could not preview data: {e}")
            print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='List available integrated spectrum files'
    )
    parser.add_argument('directories', nargs='*', default=['DSSF_CZO', 'DSSF_CZO_X'],
                       help='Directories to check (default: DSSF_CZO DSSF_CZO_X)')
    
    args = parser.parse_args()
    
    for directory in args.directories:
        list_spectrum_files(directory)


if __name__ == "__main__":
    main()
