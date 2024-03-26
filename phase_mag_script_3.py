import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *
from observables import *



# findPhaseMag110(-0.05, 0.05, 100, 0, 0.3, 100, h110, 30, 2, "phase_110_kappa=2_zoomed_in_more")
# findPhaseMag110_ex(-0.05, 0.05, 100, 0, 0.3, 100, h110, 17, 2, "phase_110_kappa=2_zoomed_in_more_ex")

Jpm = -0.2
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 40, h111, np.zeros(4),30, "Files/SSSF/Jpm=-0.2_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 40, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.2_pi")

DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.zeros(4),30, "Files/DSSF/Jpm=-0.2_0")
DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.2_pi")
