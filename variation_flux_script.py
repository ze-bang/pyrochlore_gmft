import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from variation_flux import *
from phase_diagram import *


completeSpan(-0.1,0.1,100,0,0.3,100,h110,30,2,np.zeros(4),'HanYan_110_Jpm_-0.1_0.1_h_0_0.3_0_flux')
completeSpan(-0.1,0.1,100,0,0.3,100,h110,30,2,np.ones(4)*np.pi,'HanYan_110_Jpm_-0.1_0.1_h_0_0.3_pi_flux')
completeSpan(-0.1,0.1,100,0,0.3,100,h110,30,2,np.array([np.pi, np.pi, 0,0]),'HanYan_110_Jpm_-0.1_0.1_h_0_0.3_pipi00')
completeSpan(-0.1,0.1,100,0,0.3,100,h110,30,2,np.array([0,0,np.pi,np.pi]),'HanYan_Jpm_110_-0.1_0.1_h_0_0.3_0_00pipi')

completeSpan(-0.1,0.1,100,0,0.3,100,h001,30,2,np.zeros(4),'HanYan_100_Jpm_-0.1_0.1_h_0_0.3_0_flux')
completeSpan(-0.1,0.1,100,0,0.3,100,h001,30,2,np.ones(4)*np.pi,'HanYan_100_Jpm_-0.1_0.1_h_0_0.3_pi_flux')
completeSpan(-0.1,0.1,100,0,0.3,100,h001,30,2,np.array([np.pi, 0,0, np.pi]),'HanYan_100_Jpm_-0.1_0.1_h_0_0.3_0pipi0')
completeSpan(-0.1,0.1,100,0,0.3,100,h001,30,2,np.array([0,np.pi,np.pi,0]),'HanYan_100_Jpm_-0.1_0.1_h_0_0.3_0_pi00pi')

completeSpan(-0.1,0.1,100,0,0.3,100,h111,30,2,np.zeros(4),'HanYan_111_Jpm_-0.1_0.1_h_0_0.3_0_flux')
completeSpan(-0.1,0.1,100,0,0.3,100,h111,30,2,np.ones(4)*np.pi,'HanYan_111_Jpm_-0.1_0.1_h_0_0.3_pi_flux')