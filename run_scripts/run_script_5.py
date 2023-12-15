import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *
from variation_flux import *



plot_MFE_flux_110(-0.005, -0.005, 1, 0.3, h110, 2, 26, 40, "h110_flux_plane.txt")
plot_MFE_flux_110(0.1, 0.1, 1, 0.3, h110, 2, 26, 40, "h110_flux_plane.txt")
plot_MFE_flux_110(-0.08, -0.08, 1, 0.3, h110, 2, 26, 40, "h110_flux_plane.txt")
