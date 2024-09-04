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
# plotLinefromnetCDF(h111, "h111=0.2_largeN.pdf", h=0.2, diff=True
# plotLinefromnetCDF(h111, "h111=0.3_largeN.pdf", h=0.3, diff=True)
#
# plotLinefromnetCDF(h110, "h110=0.1_largeN.pdf", h=0.1, diff=True)
# plotLinefromnetCDF(h110, "h110=0.2_largeN.pdf", h=0.2, diff=True)
# plotLinefromnetCDF(h110, "h110=0.3_largeN.pdf", h=0.3, diff=True)
#
# plotLinefromnetCDF(h001, "h001=0.1_largeN.pdf", h=0.1, diff=True)
# plotLinefromnetCDF(h001, "h001=0.2_largeN.pdf", h=0.2, diff=True)
# plotLinefromnetCDF(h001, "h001=0.3_largeN.pdf", h=0.3, diff=True)

Jpm = 0.03
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

# findXYZPhase(-1,1,-1,1,100,30,2,'XYZ_phase_diagram')
# DSSF(3, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.zeros(4), 30, "test")

# Jpm = 0.02
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h110=0")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h110=0.1")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h110=0.2")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h110=0.3")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.4, h110, np.zeros(4), 30, "Files/DSSF/Jpm=0.02/h110=0.4")

# DSSF_pedantic(200,-2*Jpm, -2*Jpm, 1, 0.35, h111, np.zeros(4), 30, "Files/DSSF/Jpm=0.03_0/h111/h=0.35")
# DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h111, np.zeros(4), 30, "Files/DSSF/Jpm=0.03_0")
# DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=0.03_pi")
#
# # DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.14, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.14")
# # DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.16, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.16")
#
# Jpm=0.03
#
# SSSF_line_pedantic(200, -2*Jpm, -2*Jpm, 1, 0, 0.4, 11, h111, np.zeros(4),30, "Files/SSSF/Jpm=0.03_0", "hh2k", 0, 3, 3)
# SSSF_line_pedantic(200, -2*Jpm, -2*Jpm, 1, 0, 0.4, 11, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.03_pi", "hh2k", 0, 3, 3)
# Jxx = np.linspace(0,0.5,1 0)
# for i in range(10):


# SSSF_pedantic(100, 0.5, 1, 0.1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/XYZ/Jpm=0.15_Jpmpm=0.1_fictitious_octupolar", "hnhl", K=0, Hr=2.5, Lr=2.5, g=0.02)
# SSSF_pedantic(100, 1, 0.5, 0.1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/XYZ/Jpm=0.15_Jpmpm=0.1_fictitious_dipolar", "hnhl", K=0, Hr=2.5, Lr=2.5, g=0.02)
findPhaseMag111(-0.3, 0.1, 40, 0, 0.4, 30, h111, 30, 2, "phase_111_kappa=2_Jpmpm=0.2", Jpmpm=0.2)
