import numpy as np

from phase_diagram import *
from observables import *
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
    rangem = 2.5
elif field_dir == "111":
    h_dir = h111
    scatplane = "hh2k"
    rangem = 3
else:
    h_dir = h001
    scatplane = "hk0"
    rangem = 2.5

if flux_in == "0":
    flux = np.zeros(4)
elif flux_in == "pi":
    flux = np.ones(4)*np.pi
else:
    flux = np.array([0,np.pi,np.pi,0])


SSSF_pedantic(100, Jxx, Jyy, Jzz, h, h_dir, flux, 30, filename, scatplane, 0, rangem, rangem, theta=theta)
