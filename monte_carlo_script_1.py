import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# from monte_carlo import *
from phase_diagram import *
from observables import *

# scan_all(h111, 4)
# scan_all(h110, 2)
# scan_all(h001, 4)

# completeSpan_ex(-0.3,0.1,100,0,1, 100,h110,17,2,np.zeros(4),'HanYan_110_Jpm_-0.3_0.1_h_0_1_0_flux_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h110,17,2,np.ones(4)*np.pi,'HanYan_110_Jpm_-0.3_0.1_h_0_1_pi_flux_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h110,17,2,np.array([np.pi, np.pi, 0,0]),'HanYan_110_Jpm_-0.3_0.1_h_0_1_pipi00_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h110,17,2,np.array([0,0,np.pi,np.pi]),'HanYan_Jpm_110_-0.3_0.1_h_0_1_0_00pipi_ex')

# Jpm = -0.05
# DSSF(0.01, -2*Jpm, -2*Jpm, 1, 0, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.05/h110=0_00pp")
# DSSF(0.01, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.3_pi_flux")
# DSSF(0.01, -2*Jpm, -2*Jpm, 1, 0.4, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.05/h110=0.4_pi_flux")

# DSSF(0.01, 0.062/0.063, 0.011/0.063, 1, 0, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jxx=0.011_Jyy=0.063_Jzz=0.062_h110=0")
# DSSF(0.01, 0.062/0.063, 0.011/0.063, 1, 0.15, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jxx=0.011_Jyy=0.063_Jzz=0.062_h110=0.15")
# DSSF(0.01, 0.062/0.063, 0.011/0.063, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jxx=0.011_Jyy=0.063_Jzz=0.062_h110=0.1")

# SSSF_HHKnK_L_integrated(100, 0.011/0.063, 0.062/0.063, 1, 0, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/Jxx=0.11_Jyy=0.062_Jzz=0.063_h110=0_HHKnK_L_integrated")
# SSSF_HHKnK_L_integrated(80, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.05, h110, np.ones(4)*np.pi, 1.25, 1.75, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.063_Jyy=0.062_Jzz=0.011_h110=0.05_HHKnK_L_integrated", 1.5, 0.3)
# SSSF_HHKnK_L_integrated(80, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.15, h110, np.ones(4)*np.pi, 1.25, 1.75, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.063_Jyy=0.062_Jzz=0.011_h110=0.15_HHKnK_L_integrated", 1.5, 0.3)


Jpm=0
SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.zeros(4), 30, "Files/Jpm=0_h110=0.1_0_flux", "hhl")
SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.zeros(4), 30, "Files/Jpm=0_h110=0.2_0_flux", "hhl")
SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.zeros(4), 30, "Files/Jpm=0_h110=0.3_0_flux", "hhl")
SSSF(100, -2*Jpm, -2*Jpm, 1, 0.4, h110, np.zeros(4), 30, "Files/Jpm=0_h110=0.4_0_flux", "hhl")

# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h001, np.zeros(4), 30, "Files/Jpm=-0.03_h001=0.1_HK0_0_flux", "hk0", 0)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h001, np.zeros(4), 30, "Files/Jpm=-0.03_h001=0.2_HK0_0_flux", "hk0", 0)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h001, np.zeros(4), 30, "Files/Jpm=-0.03_h001=0.3_HK0_0_flux", "hk0", 0)

# SSSF_HK0_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.1, h001, np.ones(4)*np.pi, -0.1, 0.1, 25, 30, "Files/Jpm=-0.03_h001=0.1_HK0_pi_flux")
# SSSF_HK0_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.2, h001, np.ones(4)*np.pi, -0.1, 0.1, 25, 30, "Files/Jpm=-0.03_h001=0.2_HK0_pi_flux")
# SSSF_HK0_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.3, h001, np.ones(4)*np.pi, -0.1, 0.1, 25, 30, "Files/Jpm=-0.03_h001=0.3_HK0_pi_flux")
