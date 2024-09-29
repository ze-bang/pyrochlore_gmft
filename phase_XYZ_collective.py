import os

import numpy as np

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
import sys

Jpm, Jpmax, Jpm1, Jpm1max, nK, nS, flux_string, filename = sys.argv[1:]
flux = ""
if flux_string == "0":
    flux = np.zeros(4)
elif flux_string == "pi":
    flux = np.ones(4)*np.pi
else:
    flux = zppz

findXYZPhase_separate(Jpm, Jpmax, Jpm1, Jpm1max, nK, 30, 2, flux, filename, nS, symmetrized=False)
