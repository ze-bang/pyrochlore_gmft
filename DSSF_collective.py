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


DSSF_pedantic(200,-2*Jpmpm - 2*Jpmpm, 1, -2*Jpmpm + 2*Jpmpm, h, h_dir, flux, 30, filename)
