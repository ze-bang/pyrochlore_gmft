import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.phase_diagram import *
from core.observables import *
import sys


Jxx, Jyy, Jzz, theta, h, field_dir, flux_in, filename = sys.argv[1:]

Jxx = float(Jxx)
Jyy = float(Jyy)
Jzz = float(Jzz)
h = float(h)
theta = float(theta)

if field_dir == "110":
    h_dir = h110
    scatplane = "hnhl"
elif field_dir == "111":
    h_dir = h111
    scatplane = "hh2k"
else:
    h_dir = h001
    scatplane = "hk0"

if flux_in == "0":
    flux = np.zeros(4)
elif flux_in == "pi":
    flux = np.ones(4)*np.pi
else:
    flux = np.array([0,np.pi,np.pi,0])

# print(Jpm, Jpmpm, h, h_dir, flux)

DSSF_pedantic(200, Jxx, Jyy, Jzz, h, h_dir, flux, 18, filename, theta=theta)
