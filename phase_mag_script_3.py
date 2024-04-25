import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *
from observables import *



# findPhaseMag110(-0.05, 0.05, 100, 0, 0.3, 100, h110, 30, 2, "phase_110_kappa=2_zoomed_in_more")
# findPhaseMag110_ex(-0.05, 0.05, 100, 0, 0.3, 100, h110, 17, 2, "phase_110_kappa=2_zoomed_in_more_ex")

# SSSF_line(100, 3001/0.063, 0.062/0.063, 1, 0, 0.3, 10, h110, np.zeros(4),30, "Files/SSSF/Jxx_0.062_Jyy_0.063_Jzz_3001_0")
# SSSF_line(100, 3001/0.063, 0.062/0.063, 1, 0, 0.3, 10, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jxx_0.062_Jyy_0.063_Jzz_3001_00pp")
# SSSF_line(100, 3001/0.063, 0.062/0.063, 1, 0, 0.3, 10, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jxx_0.062_Jyy_0.063_Jzz_3001_pi")

Jpm = -0.289

# DSSF(300, -2*Jpm, -2*Jpm, 1, 0, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289/h110=0")
# DSSF(300, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289/h110=0.15")
DSSF(300, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289/h110=0.2")
Jpm = -0.03

DSSF(300, -2*Jpm, -2*Jpm, 1, 0.15, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03/h110=0.1")
DSSF(300, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03/h110=0.2")
DSSF(300, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.03/h110=0.3")
DSSF(300, -2*Jpm, -2*Jpm, 1, 0.4, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.03/h110=0.4")
