import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *
from observables import *
# flux = np.array([np.pi, np.pi, 0, 0])

# findPhaseMag110(-0.5, 0.1, 100, 0, 2, 100, h110, 30, 2, "phase_110_kappa=2")
# findPhaseMag110_ex(-0.5, 0.1, 100, 0, 1, 100, h110, 17, 2, "phase_110_kappa=2_ex")

Jpm = 0.02
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h110, np.zeros(4),30, "Files/SSSF/Jpm=0.02_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.02_pi")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=0.02_00pp")

DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0, h111, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h111=0")
DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.15, h111, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h111=0.15")
DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.3, h111, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h111=0.3")
