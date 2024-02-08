import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *

flux = np.array([np.pi, np.pi, 0, 0])

# findPhaseMag(-0.5, 0.1, 100, 0, 3, 100, h110, 40, 2, flux, "phase_110_kappa=2")
findPhaseMag_alt(-0.5, 0.1, 100, 0, 3, 100, h110, 40, 2, flux, "phase_110_kappa=2_alt")
# findPhaseMag(-0.5, 0.1, 100, 0, 1, 100, h001, 26, 2, flux, "phase_001_kappa=2")
# findPhaseMag(-0.5, 0.1, 100, 0, 1, 100, h111, 26, 2, flux, "phase_111_kappa=2")
# findPhaseMag(-0.5, 0.1, 100, 0, 3, 100, h1b10, 26, 2, flux, "phase_1b10_kappa=2")