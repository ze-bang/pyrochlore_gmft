import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.variation_flux import *
from core.phase_diagram import *
from core.observables import *

Jpm = -0.289
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.1, 11, h001, np.zeros(4),30, "Files/SSSF/Jpm=-0.289_0", "hk0", 0)
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.1, 11, h001, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.289_pi", "hk0", 0)

# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.1, 5, h001, np.zeros(4), 30, "Files/DSSF/Jpm=-0.289_0")
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.1, 5, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_pi")
# SSSF_pedantic(100, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0.1, h111, np.ones(4)*np.pi, 30, "Files/XYZ/Ce2Zr2O7_h111=0.1", "hh2k")
# findXYZPhase_separate(-1, 1, -1, 1, 40, 30, 2, np.ones(4)*np.pi, "phase_XYZ_0_field_pi_flux_nS=1", 1)

# Jpm = 0.0
# SSSF_line_pedantic(100, -2*Jpm, 1, -2*Jpm, 0, 0.6, 11, h110, np.zeros(4),30, "Files/SSSF/Jpm=0.0_0", "hnhl")
# SSSF_line_pedantic(100, -2*Jpm, 1, -2*Jpm, 0, 0.6, 11, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.0_pi", "hnhl")
# SSSF_line_pedantic(100, -2*Jpm, 1, -2*Jpm, 0, 0.6, 11, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=0.0_00pp", "hnhl")