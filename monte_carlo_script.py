import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# from monte_carlo import *
from phase_diagram import *
from observables import *
# scan_all(h111, 2)
# scan_all(h110, 2)
# scan_all(h001, 2)

# completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.zeros(4),'HanYan_100_Jpm_-0.3_0.1_h_0_1_0_flux_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.ones(4)*np.pi,'HanYan_100_Jpm_-0.3_0.1_h_0_1_pi_flux_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.array([np.pi, 0,0, np.pi]),'HanYan_100_Jpm_-0.3_0.1_h_0_1_0pipi0_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.array([0,np.pi,np.pi,0]),'HanYan_100_Jpm_-0.3_0.1_h_0_1_0_pi00pi_ex')

Jpm = -0.03

# SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 1.25, 1.75, 25, 30, "Files/Jpm=-0.03_h110=0.1_HHKnK_pi_flux")
# SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 1.25, 1.75, 25, 30, "Files/Jpm=-0.03_h110=0.2_HHKnK_pi_flux")
# SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.ones(4)*np.pi, 1.25, 1.75, 25, 30, "Files/Jpm=-0.03_h110=0.3_HHKnK_pi_flux")

DSSF(0.005, -2*Jpm, -2*Jpm, 1, 0.4, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_h110=0.4_pi_flux")


# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h110=0.1_HHKnK_pi_flux", "hkk", 1.5)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h110=0.2_HHKnK_pi_flux", "hkk", 1.5)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h110=0.3_HHKnK_pi_flux", "hkk", 1.5)

# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h001, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h001=0.1_HK0_pi_flux", "hk0", 0)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h001, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h001=0.2_HK0_pi_flux", "hk0", 0)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h001, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h001=0.3_HK0_pi_flux", "hk0", 0)