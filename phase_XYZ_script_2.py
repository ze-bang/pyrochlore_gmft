import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from observables import *
# findXYZPhase(0.2, 0.8, 0, 0.2, 40, 30, 2, "XYZ_0_field_upper_quadrant_small_corridor")

# findPhaseMag001(-0.05, 0.05, 100, 0, 0.3, 100, h001, 30, 2, "phase_100_kappa=2_zoomed_in_more")
# findPhaseMag001_ex(-0.05, 0.05, 100, 0, 0.3, 100, h001, 17, 2, "phase_100_kappa=2_zoomed_in_more_ex")

Jpm = -0.289
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 10, h110, np.zeros(4),30, "Files/SSSF/Jpm=-0.289_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 10, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.289_pi")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 10, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=-0.289_00pp")
#
DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_h111=0")
DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.15, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_h111=0.15")
DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.3, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_h111=0.3")
