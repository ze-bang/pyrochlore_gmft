import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from archive.phase_diagram import *

findXYZPhase(0, 1, 0, 1, 40, 30, 2, "XYZ_0_field_upper_quadrant")


