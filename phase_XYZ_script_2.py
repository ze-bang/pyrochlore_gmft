import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from observables import *
Jpm = -0.289
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 30, h110, np.zeros(4),30, "Files/SSSF/Jpm=-0.289_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 30, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.289_pi")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 30, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=-0.289_00pp")
#
DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.zeros(4), 30, "Files/DSSF/Jpm=-0.289_0")
DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.289_00pp")
DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_pi")
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 11, h110, np.zeros(4),30, "Files/SSSF/Jpm=-0.289_0", "hnhl")
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 11, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.289_pi", "hnhl")
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 11, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=-0.289_00pp", "hnhl")

# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.zeros(4), 30, "Files/DSSF/Jpm=-0.289_0")
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.289_00pp")
# DSSF_line_pedantic(250,-2*Jpm, -2*Jpm, 1, 0, 0.6, 7, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.289_pi")
