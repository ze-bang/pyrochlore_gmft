from monte_carlo import *
from phase_diagram import *

# scan_all(h111, 4)
# scan_all(h110, 4)
# scan_all(h001, 2)

completeSpan_ex(-0.3,0.1,100,0,1,100,h111,30,2,np.zeros(4),'HanYan_111_Jpm_-0.3_0.1_h_0_1_0_flux_ex')
completeSpan_ex(-0.3,0.1,100,0,1,100,h111,30,2,np.ones(4)*np.pi,'HanYan_111_Jpm_-0.3_0.1_h_0_1_pi_flux_ex')
