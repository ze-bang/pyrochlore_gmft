import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *
from observables import *
# flux = np.array([np.pi, np.pi, 0, 0])

# findPhaseMag110(-0.5, 0.1, 100, 0, 2, 100, h110, 30, 2, "phase_110_kappa=2")
# findPhaseMag110_ex(-0.5, 0.1, 100, 0, 1, 100, h110, 17, 2, "phase_110_kappa=2_ex")

Jpm = 0.02
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 3, h110, np.zeros(4),30, "Files/SSSF/Jpm=0.02_0", "hhl")
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 3, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.02_pi", "hhl")
SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.3, 3, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/Jpm=0.02_00pp", "hhl")

# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0, h111, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h111=0")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.15, h111, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h111=0.15")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.3, h111, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h111=0.3")
# SSSF(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.05, h110, np.ones(4)*np.pi, 30, 'Files/SSSF/Jxx_0.063_Jyy_0.062_Jzz_0.011_h110=0.05_hhl', "hhl")
# SSSF(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.15, h110, np.ones(4)*np.pi, 30, 'Files/SSSF/Jxx_0.063_Jyy_0.062_Jzz_0.011_h110=0.15_hhl', "hhl")
