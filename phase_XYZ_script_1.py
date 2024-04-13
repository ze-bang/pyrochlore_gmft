import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
import time
from observables import *


# findPhaseMag001(-0.5, 0.1, 100, 0, 1, 100, h001, 30, 2, "phase_100_kappa=2")
# findPhaseMag001_ex(-0.5, 0.1, 100, 0, 1, 100, h001, 17, 2, "phase_100_kappa=2_ex")

Jpm = -0.05
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h110, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=-0.03_00pp")

DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h111=0")
DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.2, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h111=0.2")
DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.3, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h111=0.3")
DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.5, h111, np.zeros(4), 30, "Files/DSSF/Jpm=-0.05/h111=0.5")
