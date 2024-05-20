import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *
from observables import *


Jpm = -0.289


SSSF_line_pedantic(200, -2*Jpm, -2*Jpm, 1, 0, 0.4, 11, h111, np.zeros(4),30, "Files/SSSF/Jpm=-0.289_0", "hh2k", 0, 3, 3)
SSSF_line_pedantic(200, -2*Jpm, -2*Jpm, 1, 0, 0.4, 11, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.289_pi", "hh2k", 0, 3, 3)



<<<<<<< HEAD
DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h111, np.zeros(4), 30, "Files/DSSF/Jpm=-0.289_0")
DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_pi")
# Jpm = -0.05

# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.1")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.2")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.05/h110=0.3")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.4, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.05/h110=0.4")
# SSSF_HHL_KK_integrated(100, 0.063, 0.062, 0.011, 0, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.063_Jyy=0.062_Jzz=0.011_h110=0_HHL_KnK_integrated")
# SSSF_HHL_KK_integrated(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.1, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.063_Jyy=0.062_Jzz=0.011_h110=0.1_HHL_KnK_integrated")
# SSSF_HHL_KK_integrated(100, 0.062/0.063, 0.063/0.063, 0.011/0.063, 1.5, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.11_Jyy=0.062_Jzz=0.063_h110=0.1_HHL_KnK_integrated")

# findPhaseMag110(-0.5, 0.1, 100, 0, 2, 100, h110, 30, 2, "phase_110_kappa=2_dipolar", Jxx=True)
# findPhaseMag111(-0.5, 0.1, 100, 0, 1, 100, h111, 30, 2, "phase_111_kappa=2_dipolar", Jxx=True)
# findPhaseMag001(-0.5, 0.1, 100, 0, 1, 100, h001, 30, 2, "phase_100_kappa=2_dipolar", Jxx=True)
=======
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h111, np.zeros(4), 30, "Files/DSSF/Jpm=-0.289_0")
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_pi")
>>>>>>> 3ba3618cd8a7545c8913cb413fd8d86dee0d6642
