import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
import time
from observables import *

Jpm = -0.03
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.6, 11, h110, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0", "hnhl")
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.6, 11, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi", "hnhl")
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.6, 11, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=-0.03_00pp", "hnhl")

# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.zeros(4), 30, "Files/DSSF/Jpm=-0.03_0")
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.03_00pp")
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_pi")