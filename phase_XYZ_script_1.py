import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
import time
from observables import *


# findPhaseMag001(-0.5, 0.1, 100, 0, 1, 100, h001, 30, 2, "phase_100_kappa=2")
# findPhaseMag001_ex(-0.5, 0.1, 100, 0, 1, 100, h001, 17, 2, "phase_100_kappa=2_ex")

Jpm = -0.03
SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 40, h110, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0")
SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 40, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi")
SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 40, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=-0.03_00pp")

# DSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 20, h110, np.zeros(4),30, "Files/DSSF/Jpm=-0.03_0")
# DSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 20, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_pi")
# DSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 20, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.03_pi_00pp")
