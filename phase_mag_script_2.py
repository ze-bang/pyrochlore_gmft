import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *
from flux_stuff import *


findPhaseMag111(-0.05, 0.05, 100, 0, 0.3, 100, h111, 30, 2, "phase_111_kappa=2_zoomed_in_more")
# findPhaseMag111_ex(-0.05, 0.05, 100, 0, 0.3, 100, h111, 17, 2, "phase_111_kappa=2_zoomed_in_more_ex")

