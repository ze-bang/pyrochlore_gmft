import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# from monte_carlo import *
from phase_diagram import *
from observables import *


# scan_all(h111, 4)
# scan_all(h110, 4)
# scan_all(h001, 2)

# completeSpan_ex(-0.3,0.1,100,0,1,100,h111,17,2,np.zeros(4),'HanYan_111_Jpm_-0.3_0.1_h_0_1_0_flux_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h111,17,2,np.ones(4)*np.pi,'HanYan_111_Jpm_-0.3_0.1_h_0_1_pi_flux_ex')

# Jpm = -0.05

# # SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/Jpm=-0.03_h110=0.1_HHKnK_00pp_flux", "hkk", 1.5)
# # SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/Jpm=-0.03_h110=0.2_HHKnK_00pp_flux", "hkk", 1.5)
# # SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/Jpm=-0.03_h110=0.3_HHKnK_00pp_flux", "hkk", 1.5)

# # SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 1.25, 1.75, 51, 30, "Files/Jpm=-0.05_h110=0.1_HHKnK_L_integrated")
# # SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 1.25, 1.75, 51, 30, "Files/Jpm=-0.05_h110=0.2_HHKnK_L_integrated")
# # SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.ones(4)*np.pi, 1.25, 1.75, 51, 30, "Files/Jpm=-0.05_h110=0.3_HHKnK_L_integrated")
# # SSSF_HHKnK_L_integrated(80, 0.062, 0.063, 0.011, 1.5, h110, np.ones(4)*np.pi, 1.25, 1.75, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.11_Jyy=0.062_Jzz=0.063_B110=1.5_HHKnK_L_integrated", 1.5, 0.3)
# Jpm=0
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/Jpm=0_h110=0.1_pi_flux", "hhl")
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 30, "Files/Jpm=0_h110=0.2_pi_flux", "hhl")
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.ones(4)*np.pi, 30, "Files/Jpm=0_h110=0.3_pi_flux", "hhl")
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.4, h110, np.ones(4)*np.pi, 30, "Files/Jpm=0_h110=0.4_pi_flux", "hhl")
findXYZPhase_separate(-0.5, 1, -0.5, 1, 80, 30, 2, np.ones(4)*np.pi, "phase_XYZ_0_field_pi_flux_nS=1_detailed", 1)
