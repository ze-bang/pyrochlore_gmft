import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
from misc_helper import *
from flux_stuff import *
from observables import *

# findPhaseMag111(-0.05, 0.05, 100, 0, 0.3, 100, h111, 30, 2, "phase_111_kappa=2_zoomed_in_more")
# findPhaseMag111_ex(-0.05, 0.05, 100, 0, 0.3, 100, h111, 17, 2, "phase_111_kappa=2_zoomed_in_more_ex")


# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi")

# SSSF(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.05, h110, np.ones(4)*np.pi, 30, 'Files/SSSF/Jxx_0.063_Jyy_0.062_Jzz_0.011_h110=0.05_hhk_1.5_zoomed_in', "hkk", 1.5, 1.5, 0.3)
# SSSF(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.15, h110, np.ones(4)*np.pi, 30, 'Files/SSSF/Jxx_0.063_Jyy_0.062_Jzz_0.011_h110=0.15_hhk_1.5_zoomed_in', "hkk", 1.5, 1.5, 0.3)
# SSSF(100, 0.011/0.063, 0.062/0.063, 1, 0.18, h110, np.ones(4)*np.pi, 30, 'Jxx_0.062_Jyy_0.063_Jzz_0.011_h110=0.2', "hhl")


# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 3, h111, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 3, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi")

# DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.zeros(4),30, "Files/DSSF/Jpm=-0.03_0")
# DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_pi")

# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.02, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.02")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.04, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.04")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.06, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.06")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.08, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.05/h110=0.08")
# DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.12, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.05/h110=0.12")
