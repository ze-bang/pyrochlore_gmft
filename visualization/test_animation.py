#!/usr/bin/env python3
"""
Quick test script to generate an animation from existing data files.
This is a minimal example showing how to use the animation feature.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualization.animate_spectrum import animate_integrated_spectrum
import os

# Check if directories exist
if os.path.exists('DSSF_CZO'):
    print("Creating animation for DSSF_CZO directory...")
    animate_integrated_spectrum(
        dir='DSSF_CZO',
        H_min=0,
        H_max=4,
        n_H=20,
        output_filename='test_animation_CZO.gif',
        fps=5,
        dpi=100
    )
    print("\nDone! Check test_animation_CZO.gif")
else:
    print("DSSF_CZO directory not found!")

if os.path.exists('DSSF_CZO_X'):
    print("\nCreating animation for DSSF_CZO_X directory...")
    animate_integrated_spectrum(
        dir='DSSF_CZO_X',
        H_min=0,
        H_max=4,
        n_H=20,
        output_filename='test_animation_CZO_X.gif',
        fps=5,
        dpi=100
    )
    print("\nDone! Check test_animation_CZO_X.gif")
else:
    print("DSSF_CZO_X directory not found!")
