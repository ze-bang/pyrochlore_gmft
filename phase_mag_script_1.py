import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# import numpy as np
from phase_diagram import *
from misc_helper import *
from flux_stuff import *
from observables import *



# findPhaseMag111_ex(-0.5, 0.1, 100, 0, 1, 100, h111, 17, 2, "phase_111_kappa=2_ex")
# findPhaseMag111(-0.5, 0.1, 100, 0, 1, 100, h111, 30, 2, "phase_111_kappa=2")

# plotLinefromnetCDF(h111, "h111=0.1_largeN.pdf", h=0.1, diff=True)
# plotLinefromnetCDF(h111, "h111=0.2_largeN.pdf", h=0.2, diff=True)
# plotLinefromnetCDF(h111, "h111=0.3_largeN.pdf", h=0.3, diff=True)
#
# plotLinefromnetCDF(h110, "h110=0.1_largeN.pdf", h=0.1, diff=True)
# plotLinefromnetCDF(h110, "h110=0.2_largeN.pdf", h=0.2, diff=True)
# plotLinefromnetCDF(h110, "h110=0.3_largeN.pdf", h=0.3, diff=True)
#
# plotLinefromnetCDF(h001, "h001=0.1_largeN.pdf", h=0.1, diff=True)
# plotLinefromnetCDF(h001, "h001=0.2_largeN.pdf", h=0.2, diff=True)
# plotLinefromnetCDF(h001, "h001=0.3_largeN.pdf", h=0.3, diff=True)

Jpm = 0.02
# py0s = pycon.piFluxSolver(-2*Jpm, -2*Jpm, 1, BZres=25, h=0.3, n=h111, flux=np.ones(4)*np.pi)
# py0s.solvemeanfield()
# print(SSSF_core(np.array([0.5,1,0.5]), hb110, py0s))
# print(SSSF_core(np.array([1,1,1]), hb110, py0s))
#
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.ones(4)*np.pi,30, "test")
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi,30, "test", "hhl")

#
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.zeros(4),30, "Files/SSSF/Jpm=0.02_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.02_pi")

# DSSF(0.01, -2*Jpm, -2*Jpm, 1, 0, h110, np.zeros(4), 10, "Files/DSSF/Jpm=0.02_h110=0")
DSSF(0.01, -2*Jpm, -2*Jpm, 1, 0.05, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02_h110=0.15_r")
DSSF(0.01, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02_h110=0.4_r")

# SSSF_HHL_KK_integrated(10, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, -0.3, 0.3, 3, 5, 'Jpm=-0.289_h110=0.2_L_integrated')

# DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.zeros(4),30, "Files/DSSF/Jpm=0.02_0")
# DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=0.02_pi")
