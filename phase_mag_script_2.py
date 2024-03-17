import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *
from flux_stuff import *
from observables import *

# findPhaseMag111(-0.05, 0.05, 100, 0, 0.3, 100, h111, 30, 2, "phase_111_kappa=2_zoomed_in_more")
# findPhaseMag111_ex(-0.05, 0.05, 100, 0, 0.3, 100, h111, 17, 2, "phase_111_kappa=2_zoomed_in_more_ex")

Jpm = -0.03

SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 40, h111, h110, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0")
SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 40, h111, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi")

# DSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 40, h111, np.zeros(4),30, "Files/DSSF/Jpm=-0.03_0")
# DSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 40, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_pi")
