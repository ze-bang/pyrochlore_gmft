import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *

flux = np.array([np.pi/4, -3*np.pi/4, np.pi/4, np.pi/4])


findPhaseMag(-0.5, 0.1, 100, 0, 0.3, 100, h110, 26, 2, flux, "FF_phase_110_kappa=2")
findPhaseMag(-0.5, 0.1, 100, 0, 0.3, 100, h001, 26, 2, flux, "FF_phase_001_kappa=2")
findPhaseMag(-0.5, 0.1, 100, 0, 0.3, 100, h111, 26, 2, flux, "FF_phase_111_kappa=2")
findPhaseMag(-0.5, 0.1, 100, 0, 0.3, 100, h1b10, 26, 2, flux, "FF_phase_1b10_kappa=2")
