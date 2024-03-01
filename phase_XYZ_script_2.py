import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *

# findXYZPhase(0.2, 0.8, 0, 0.2, 40, 30, 2, "XYZ_0_field_upper_quadrant_small_corridor")

# findPhaseMag001(-0.05, 0.05, 100, 0, 0.3, 100, h001, 30, 2, "phase_100_kappa=2_zoomed_in_more")
findPhaseMag001_ex(-0.05, 0.05, 100, 0, 0.3, 100, h001, 30, 2, "phase_100_kappa=2_zoomed_in_more_ex")
