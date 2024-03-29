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

Jpm = -0.03

SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.zeros(4), 30, "Files/Jpm=-0.03_h110=0.1_HHKnK_0_flux", "hkk", 1.5)
SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.zeros(4), 30, "Files/Jpm=-0.03_h110=0.2_HHKnK_0_flux", "hkk", 1.5)
SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.zeros(4), 30, "Files/Jpm=-0.03_h110=0.3_HHKnK_0_flux", "hkk", 1.5)

SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h001, np.zeros(4), 30, "Files/Jpm=-0.03_h001=0.1_HK0_0_flux", "hk0", 0)
SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h001, np.zeros(4), 30, "Files/Jpm=-0.03_h001=0.2_HK0_0_flux", "hk0", 0)
SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h001, np.zeros(4), 30, "Files/Jpm=-0.03_h001=0.3_HK0_0_flux", "hk0", 0)
