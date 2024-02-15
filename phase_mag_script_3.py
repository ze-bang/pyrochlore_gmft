import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *


flux = np.array([np.pi, np.pi, 0, 0])

# findPhaseMag(-0.1, 0.1, 100, 0, 0.3, 100, h110, 35, 2, flux, "phase_110_kappa=2_zoomed_in")
# findPhaseMag(-0.1, 0.1, 100, 0, 0.3, 100, h001, 35, 2, flux, "phase_001_kappa=2_zoomed_in")
# findPhaseMag(-0.1, 0.1, 100, 0, 0.3, 100, h111, 35, 2, flux, "phase_111_kappa=2_zoomed_in")
# findPhaseMag(-0.1, 0.1, 100, 0, 0.3, 100, h1b10, 35, 2, flux, "phase_1b10_kappa=2_zoomed_in")

# findPhaseMag(-0.05, 0.05, 100, 0, 0.3, 100, h110, 30, 2, flux, "phase_110_kappa=2_zoomed_in_more")
# findPhaseMag(-0.05, 0.05, 100, 0, 0.3, 100, h001, 26, 2, flux, "phase_001_kappa=2_zoomed_in_more")
# findPhaseMag(-0.05, 0.05, 100, 0, 0.3, 100, h111, 26, 2, flux, "phase_111_kappa=2_zoomed_in_more")
# findPhaseMag(-0.05, 0.05, 100, 0, 0.3, 100, h1b10, 26, 2, flux, "phase_1b10_kappa=2_zoomed_in_more")

# findPhaseMag(-0.05, 0.05, 100, 0, 0.3, 100, h110, 40, 2, flux, "phase_110_kappa=2_zoomed_in_more")

findPhaseMag110(-0.05, 0.05, 100, 0, 0.3, 100, h110, 30, 2, "phase_110_kappa=2_zoomed_in_more")

# findPhaseMag_simple(-0.05, 0.05, 100, 0, 0.3, 100, h001, 26, 2, flux, "phase_001_pi0")
# findPhaseMag_simple(-0.05, 0.05, 100, 0, 0.3, 100, h110, 26, 2, flux, "phase_110_pi0")
# findPhaseMag_simple(-0.05, 0.05, 100, 0, 0.3, 100, h111, 26, 2, flux, "phase_111_pi0")
#
