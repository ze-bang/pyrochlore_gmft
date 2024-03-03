import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *




# findPhaseMag110(-0.05, 0.05, 100, 0, 0.3, 100, h110, 30, 2, "phase_110_kappa=2_zoomed_in_more")
findPhaseMag110_ex(-0.05, 0.05, 100, 0, 0.3, 100, h111, 25, 2, "phase_110_kappa=2_zoomed_in_more_ex")
