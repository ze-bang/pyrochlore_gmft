import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# import numpy as np
from phase_diagram import *
from misc_helper import *
from flux_stuff import *
from observables import *


# findPhaseMag111_ex(-0.5, 0.1, 100, 0, 1, 100, h111, 17, 2, "phase_111_kappa=2_ex")
# findPhaseMag111(-0.5, 0.1, 100, 0, 1, 100, h111, 30, 2, "phase_111_kappa=2")

Jpm = -0.05
nH = 40
h = np.linspace(0, 0.5, 40)

for i in range(nH):
    SSSF(100, -2*Jpm, -2*Jpm, 1, h[i], h110, np.array([1,1,0]),np.array([0,0,np.pi,np.pi]),30,"SSSF_00pp_Jpm=-0.03_h110="+str(h[i]))
    SSSF(100, -2*Jpm, -2*Jpm, 1, h[i], h110, np.array([1,1,0]),np.array([0,0,0,0]),30,"SSSF_0_Jpm=-0.03_h110=0.5"+str(h[i]))
    SSSF(100, -2*Jpm, -2*Jpm, 1, h[i], h110, np.array([1,1,0]),np.array([np.pi,np.pi,np.pi,np.pi]),30,"SSSF_pi_Jpm=-0.03_h110=0.5"+str(h[i]))