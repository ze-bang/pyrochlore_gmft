import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from variation_flux import *
from phase_diagram import *
from observables import *

Jpm = 0.0

# SSSF_line_pedantic(100, 1, -2*Jpm, -2*Jpm, 0, 0.4, 11, h001, np.zeros(4),30, "Files/SSSF/Jpm=0.02_0", "hk0", 0)
# SSSF_line_pedantic(100, 1, -2*Jpm, -2*Jpm, 0, 0.4, 11, h001, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.02_pi", "hk0", 0)


# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 10, h110, np.zeros(4),30, "Files/SSSF/Jpm=0.02_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 10, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.02_pi")
#
# DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h001, np.zeros(4),30, "Files/DSSF/Jpm=5e-3_0")
# DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=5e-3_pi")

# DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h001, np.zeros(4), 30, "Files/DSSF/Jpm=0.03_0")
DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=0.0_pi")
