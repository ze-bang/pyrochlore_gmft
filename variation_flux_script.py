import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from variation_flux import *
from phase_diagram import *
from observables import *

# completeSpan(-0.1,0.1,1,0.2,0.3,1,h110,5,2,np.zeros(4),'HanYan_110_Jpm_-0.1_0.1_h_0_0.3_0_flux')


# completeSpan(-0.1,0.1,100,0,0.3,100,h110,5,2,np.zeros(4),'HanYan_110_Jpm_-0.1_0.1_h_0_0.3_0_flux')
# completeSpan(-0.1,0.1,100,0,0.3,100,h110,5,2,np.ones(4)*np.pi,'HanYan_110_Jpm_-0.1_0.1_h_0_0.3_pi_flux')
# completeSpan(-0.1,0.1,100,0,0.3,100,h110,5,2,np.array([np.pi, np.pi, 0,0]),'HanYan_110_Jpm_-0.1_0.1_h_0_0.3_pipi00')
# completeSpan(-0.1,0.1,100,0,0.3,100,h110,5,2,np.array([0,0,np.pi,np.pi]),'HanYan_Jpm_110_-0.1_0.1_h_0_0.3_0_00pipi')

# completeSpan(-0.1,0.1,100,0,0.3,100,h001,30,2,np.zeros(4),'HanYan_100_Jpm_-0.1_0.1_h_0_0.3_0_flux')
# completeSpan(-0.1,0.1,100,0,0.3,100,h001,30,2,np.ones(4)*np.pi,'HanYan_100_Jpm_-0.1_0.1_h_0_0.3_pi_flux')
# completeSpan(-0.1,0.1,100,0,0.3,100,h001,30,2,np.array([np.pi, 0,0, np.pi]),'HanYan_100_Jpm_-0.1_0.1_h_0_0.3_0pipi0')
# completeSpan(-0.1,0.1,100,0,0.3,100,h001,30,2,np.array([0,np.pi,np.pi,0]),'HanYan_100_Jpm_-0.1_0.1_h_0_0.3_0_pi00pi')

# completeSpan(-0.1,0.1,100,0,0.3,100,h111,30,2,np.zeros(4),'HanYan_111_Jpm_-0.1_0.1_h_0_0.3_0_flux')
# completeSpan(-0.1,0.1,100,0,0.3,100,h111,30,2,np.ones(4)*np.pi,'HanYan_111_Jpm_-0.1_0.1_h_0_0.3_pi_flux')

Jpm = 0.02

# SSSF_line_pedantic(100, 1, -2*Jpm, -2*Jpm, 0, 0.4, 11, h001, np.zeros(4),30, "Files/SSSF/Jpm=0.02_0", "hk0", 0)
# SSSF_line_pedantic(100, 1, -2*Jpm, -2*Jpm, 0, 0.4, 11, h001, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.02_pi", "hk0", 0)
# SSSF_line(100, 1, -2*Jpm, -2*Jpm, 0, 0.2, 5, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/SSSF/dipolar/Jpm=-0.03_00pp")


# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 10, h110, np.zeros(4),30, "Files/SSSF/Jpm=0.02_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 10, h110, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=0.02_pi")
#
# DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h001, np.zeros(4),30, "Files/DSSF/Jpm=5e-3_0")
# DSSF_line(5e-3, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=5e-3_pi")

DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h001, np.zeros(4), 30, "Files/DSSF/Jpm=0.02_0")
DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h001, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=0.02_pi")
