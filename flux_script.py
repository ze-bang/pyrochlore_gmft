import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from variation_flux import *
from phase_diagram import *
from observables import *

Jpm = -0.289
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.1, 11, h001, np.zeros(4),30, "Files/SSSF/Jpm=-0.289_0", "hk0", 0)
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.1, 11, h001, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.289_pi", "hk0", 0)

# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.1, 5, h001, np.zeros(4), 30, "Files/DSSF/Jpm=-0.289_0")
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.1, 5, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_pi")
