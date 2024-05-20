import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from variation_flux import *
from phase_diagram import *
from observables import *

<<<<<<< HEAD
Jpm = -0.03
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.25, 10, h001, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.25, 10, h001, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi")
#
# DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h001, np.zeros(4),30, "Files/DSSF/Jpm=-0.03_0")
# DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_pi")

DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.2, 5, h001, np.zeros(4), 30, "Files/DSSF/Jpm=-0.03_0")
DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.2, 5, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_pi")
=======
# Jpm = -0.03
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.2, 5, h001, np.zeros(4), 30, "Files/DSSF/Jpm=-0.03_0")
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.2, 5, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_pi")
>>>>>>> 3ba3618cd8a7545c8913cb413fd8d86dee0d6642

Jpm = -0.03
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.2, 11, h001, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0", "hk0", 0)
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.2, 11, h001, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi", "hk0", 0)
