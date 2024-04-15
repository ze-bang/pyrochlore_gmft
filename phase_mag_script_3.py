import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *
from observables import *



# findPhaseMag110(-0.05, 0.05, 100, 0, 0.3, 100, h110, 30, 2, "phase_110_kappa=2_zoomed_in_more")
# findPhaseMag110_ex(-0.05, 0.05, 100, 0, 0.3, 100, h110, 17, 2, "phase_110_kappa=2_zoomed_in_more_ex")

# SSSF_line(100, 0.011/0.063, 0.062/0.063, 1, 0, 0.3, 10, h110, np.zeros(4),30, "Files/SSSF/Jxx_0.062_Jyy_0.063_Jzz_0.011_0")
# SSSF_line(100, 0.011/0.063, 0.062/0.063, 1, 0, 0.3, 10, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jxx_0.062_Jyy_0.063_Jzz_0.011_00pp")
# SSSF_line(100, 0.011/0.063, 0.062/0.063, 1, 0, 0.3, 10, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jxx_0.062_Jyy_0.063_Jzz_0.011_pi")

# Jpm = -0.289

# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289/h110=0")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289/h110=0.1")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289/h110=0.2")
# Jpm = -0.05

# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.1")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.2")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.05/h110=0.3")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.4, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.05/h110=0.4")
SSSF_HHL_KK_integrated(100, 0.011/0.063, 0.062/0.063, 1, 0, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.11_Jyy=0.062_Jzz=0.063_h110=0_HHL_KnK_integrated")
SSSF_HHL_KK_integrated(100, 0.011/0.063, 0.062/0.063, 1, 0.05, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.11_Jyy=0.062_Jzz=0.063_h110=0.05_HHL_KnK_integrated")
SSSF_HHL_KK_integrated(100, 0.011/0.063, 0.062/0.063, 1, 0.1, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.11_Jyy=0.062_Jzz=0.063_h110=0.1_HHL_KnK_integrated")
