import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
import sys

Jpm, Jpmax, Jpm1, Jpm1max, nK, nS, flux, filename = sys.argv[1:]
findXYZPhase_separate(Jpm, Jpmax, Jpm1, Jpm1max, nK, 30, 2, flux, filename, nS, symmetrized=False)
