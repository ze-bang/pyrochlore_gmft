import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *


findXYZPhase(0.2, 0.8, 0, 0.2, 40, 30, 2, "XYZ_0_field_upper_quadrant_small_corridor")

