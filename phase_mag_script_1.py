import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# import numpy as np
from phase_diagram import *
from misc_helper import *
from flux_stuff import *


findPhaseMag111(-0.5, 0.1, 100, 0, 1, 100, h111, 30, 2, "phase_111_kappa=2_ex")
# findPhaseMag111(-0.5, 0.1, 100, 0, 1, 100, h111, 30, 2, "phase_111_kappa=2")

