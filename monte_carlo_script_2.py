import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from monte_carlo import *
from phase_diagram import *
from observables import *


# scan_all(h111, 4)
# scan_all(h110, 4)
# scan_all(h001, 2)

# completeSpan_ex(-0.3,0.1,100,0,1,100,h111,17,2,np.zeros(4),'HanYan_111_Jpm_-0.3_0.1_h_0_1_0_flux_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h111,17,2,np.ones(4)*np.pi,'HanYan_111_Jpm_-0.3_0.1_h_0_1_pi_flux_ex')

Jpm = -0.03

<<<<<<< HEAD
SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.array([0,0,np.pi,np.pi]), 1.25, 1.75, 10, 30, "Files/Jpm=-0.03_h=0.1_HHKnK_00pp_flux")
SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.array([0,0,np.pi,np.pi]), 1.25, 1.75, 10, 30, "Files/Jpm=-0.03_h=0.2_HHKnK_00pp_flux")
SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.array([0,0,np.pi,np.pi]), 1.25, 1.75, 10, 30, "Files/Jpm=-0.03_h=0.3_HHKnK_00pp_flux")
=======
SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.array([0,0,np.pi,np.pi]), 1.25, 1.75, 10, 30, "Files/Jpm=-0.03_h110=0.1_HHKnK_00pp_flux")
SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.array([0,0,np.pi,np.pi]), 1.25, 1.75, 10, 30, "Files/Jpm=-0.03_h110=0.2_HHKnK_00pp_flux")
SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.array([0,0,np.pi,np.pi]), 1.25, 1.75, 10, 30, "Files/Jpm=-0.03_h110=0.3_HHKnK_00pp_flux")
>>>>>>> ca4fc74a31f7bae6daf82f8ea22e85eaaa111c2b
