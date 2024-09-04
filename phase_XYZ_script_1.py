import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
import time
from observables import *

Jpm = -0.03
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.6, 11, h110, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0", "hnhl")
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.6, 11, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi", "hnhl")
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.6, 11, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=-0.03_00pp", "hnhl")

# DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.zeros(4), 30, "Files/DSSF/Jpm=-0.03_0")
# DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.03_00pp")
# DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_pi")
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.6, 11, h110, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0", "hnhl")
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.6, 11, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi", "hnhl")
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.6, 11, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=-0.03_00pp", "hnhl")

# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h111=0")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.2, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h111=0.2")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.3, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h111=0.3")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.5, h111, np.zeros(4), 30, "Files/DSSF/Jpm=-0.05/h111=0.5")

# SSSF(100, 0.063, 0.062, 0.011, 0, h110, np.ones(4)*np.pi, 30, 'Files/SSSF/Jxx_0.063_Jyy_0.062_Jzz_0.011_h110=0_hhl', "hhl")
# SSSF(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.1, h110, np.ones(4)*np.pi, 30, 'Files/SSSF/Jxx_0.063_Jyy_0.062_Jzz_0.011_h110=0.1_hhl', "hhl")
# SSSF(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.2, h110, np.ones(4)*np.pi, 30, 'Files/SSSF/Jxx_0.063_Jyy_0.062_Jzz_0.011_h110=0.2_hhl', "hhl")
#findXYZPhase_separate_unconstrained(-1, 1, -1, 1, 100, 30, 2, np.ones(4)*np.pi, "phase_XYZ_0_field_pi_flux_unconstrained")

# findXYZPhase_separate(0, 1, 0, 1, 20, 30, 2, np.zeros(4), "phase_XYZ_0_field_0_flux_nS=1_0101", 1)
findPhaseMag110(-0.3125, 0.1, 40, 0, 0.35, 20, h110, 30, 2, "phase_110_kappa=2_Jpmpm=0.2", Jpmpm=0.1875)
