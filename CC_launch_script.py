import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from variation_flux import *
from phase_diagram import *
from observables import *

# Jpm = -0.03
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.2, 5, h001, np.zeros(4), 30, "Files/DSSF/Jpm=-0.03_0")
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.2, 5, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_pi")
# 
# Jpm = -0.03
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.2, 11, h001, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0", "hk0", 0)
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.2, 11, h001, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi", "hk0", 0)


# SSSF_pedantic(100, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0.1, h110, np.ones(4)*np.pi, 30, "Files/XYZ/Ce2Zr2O7_h110=0.1", "hnhl")
# findXYZPhase_separate(-1, 1, -1, 1, 40, 30, 2, np.zeros(4), "phase_XYZ_0_field_0_flux_nS=1",1)
Jpm=0.00

SSSF_line_pedantic(100, -2*Jpm, 1, -2*Jpm, 0, 0.4, 11, h111, np.zeros(4),30, "Files/SSSF/Jpm=0.0_0", "hh2k", 0, 3, 3)
SSSF_line_pedantic(100, -2*Jpm, 1, -2*Jpm, 0, 0.4, 11, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.0_pi", "hh2k", 0, 3, 3)