import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
import time



findPhaseMag001(-0.5, 0.1, 100, 0, 1, 100, h001, 30, 2, "phase_100_kappa=2")
# findPhaseMag001_ex(-0.5, 0.1, 100, 0, 1, 100, h001, 17, 2, "phase_100_kappa=2_ex")
