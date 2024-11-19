import numpy as np

from phase_diagram import *
from observables import *
import sys


Jpm, Jpmpm, h, field_dir, flux_in, filename = sys.argv[1:]

Jpm = float(Jpm)
Jpmpm = float(Jpmpm)
h = float(h)


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


DSSF_pedantic(200,-2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, h, h_dir, flux, 30, filename)
