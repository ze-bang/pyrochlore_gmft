import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *
from observables import *



# findPhaseMag110(-0.05, 0.05, 100, 0, 0.3, 100, h110, 30, 2, "phase_110_kappa=2_zoomed_in_more")
# findPhaseMag110_ex(-0.05, 0.05, 100, 0, 0.3, 100, h110, 17, 2, "phase_110_kappa=2_zoomed_in_more_ex")

Jpm = -0.289
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.35, 10, h111, np.zeros(4),30, "Files/SSSF/Jpm=-0.289_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.35, 10, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.289_pi")

DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_h110=0")
DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_h110=0.1")
DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_h110=0.2")
