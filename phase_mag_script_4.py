import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *

flux = np.array([np.pi, np.pi, 0, 0])

findPhaseMag110(-0.5, 0.1, 100, 0, 2, 100, h110, 30, 2, "phase_110_kappa=2")
# findPhaseMag110_ex(-0.5, 0.1, 100, 0, 1, 100, h110, 17, 2, "phase_110_kappa=2_ex")
