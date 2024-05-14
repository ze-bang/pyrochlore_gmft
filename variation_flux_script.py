import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from variation_flux import *
from phase_diagram import *
from observables import *

Jpm = 0.02

SSSF_line_pedantic(100, 1, -2*Jpm, -2*Jpm, 0, 0.4, 11, h001, np.zeros(4),30, "Files/SSSF/Jpm=0.02_0", "hk0", 0)
SSSF_line_pedantic(100, 1, -2*Jpm, -2*Jpm, 0, 0.4, 11, h001, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.02_pi", "hk0", 0)


# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h001, np.zeros(4), 30, "Files/DSSF/Jpm=0.02_0")
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=0.02_pi")
