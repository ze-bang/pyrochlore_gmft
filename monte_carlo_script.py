from monte_carlo import *
from phase_diagram import *

# scan_all(h111, 2)
# scan_all(h110, 2)
# scan_all(h001, 2)

completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.zeros(4),'HanYan_100_Jpm_-0.3_0.1_h_0_1_0_flux_ex')
completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.ones(4)*np.pi,'HanYan_100_Jpm_-0.3_0.1_h_0_1_pi_flux_ex')
completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.array([np.pi, 0,0, np.pi]),'HanYan_100_Jpm_-0.3_0.1_h_0_1_0pipi0_ex')
completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.array([0,np.pi,np.pi,0]),'HanYan_100_Jpm_-0.3_0.1_h_0_1_0_pi00pi_ex')
