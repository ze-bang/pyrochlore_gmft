import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from archive.phase_diagram import *

findXYZPhase(0.2, 0.8, 0, 0.2, 40, 30, 2, "XYZ_0_field_upper_quadrant_small_corridor")

