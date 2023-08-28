import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
import math
import time
import sys
from phase_diagram import *
from numba import jit
from misc_helper import *




#region graph dispersion
def graphdispersion(JP,h, n, kappa, rho, graphres, BZres):
    if JP >= 0:
        py0s = py0.zeroFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
        py0s.findminLam()
        py0s.findLambda()
        # print(py0s.lams, py0s.minLams, py0s.condensed())
        # plt.axvline(x=py0s.lams[0], color='b', label='axvline - full height', linestyle='dashed')
        # plt.plot(py0s.lams[0], py0s.rho(py0s.lams)[0], marker='o')
        py0s.graph(False)
        # py0s.graphAlg(False)
        # py0s.graphAlg(False)
        # plt.legend(['Num', 'Alg'])
        plt.show()
    elif JP < 0:
        py0s = pypi.piFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
        py0s.findLambda()
        # temp = py0s.M_true(py0s.bigB)[:,0:4, 0:4] - np.conj(py0s.M_true(py0s.bigB)[:,4:8, 4:8])
        py0s.graph(True)
#endregion




# graphdispersion(-1/3, 0.8, h110, 1, 2, 20, 20)
# graphdispersion(0.02,0.8, h111, 1, 2, 20, 20)
# graphdispersion(0.046, 0, h111, 1, 2, 20, 20)
# graphdispersion(0.05, 0, h111, 1, 2, 20, 20)
# graphdispersion(0.1, 0, h111, 1, 2, 20, 20)
# plt.show()

# findPhase(60,20, 20, "Files/phase_diagram.txt")

# PhaseMagtestH(0.0001, 0.25, 25, 0, 3, 25, h110, 35, 1, "0.txt")

# PhaseMagtestJP(0, 0.25, 25, 0, 3, 25, h111, 35, 1, "0.txt")
#
#


#
# # SSSF(25, 0, np.array([1,1,1]),0.02,25, "SSSF_zero_0.02_h111=0")
#
# # SSSF(25, 0, h111,0.06,25, "SSSF_zero_0.06")

# SSSF(25, 0, np.array([1,1,1]),0.04,10, "SSSF_zero_test")
#
#
# SSSF(100, 0, hb110,-0.05,30, "SSSF_pi_-0.05_DETAILED")
# SSSF(100, 0, hb110,-0.20,30, "SSSF_pi_-0.20_DETAILED")
# SSSF(100, 0, hb110,-0.40,30, "SSSF_pi_-0.40_DETAILED")
# SSSF(100, 0, hb110,0.02,30, "SSSF_pi_0.02_DETAILED")
# SSSF(100, 0, hb110,0.03,30, "SSSF_pi_0.03_DETAILED")
# SSSF(100, 0, hb110,0.04,30, "SSSF_pi_0.04_DETAILED")



# SSSF(50, 0, np.array([1,1,1]),-0.40,30, "SSSF_pi_-0.40_DETAILED")

# SSSF(25, 0, h111,0.04,10, "SSSF_zero_0.02")

SSSF(50, 0.2, h111,-1/3,30, "SSSF_pi_-0.33_h111=0.2")
SSSF(50, 0.3, h111,-1/3,30, "SSSF_pi_-0.33_h111=0.3")
SSSF(50, 0.4, h111,-1/3,30, "SSSF_pi_-0.33_h111=0.4")

SSSF(50, 0.2, h001,-1/3,30, "SSSF_pi_-0.33_h001=0.2")
SSSF(50, 0.3, h001,-1/3,30, "SSSF_pi_-0.33_h001=0.3")
SSSF(50, 0.4, h001,-1/3,30, "SSSF_pi_-0.33_h001=0.4")

SSSF(50, 0.8, h110,-1/3,30, "SSSF_pi_-0.33_h110=0.8")
SSSF(50, 1.2, h110,-1/3,30, "SSSF_pi_-0.33_h110=1.2")
SSSF(50, 1.6, h110,-1/3,30, "SSSF_pi_-0.33_h110=1.6")

SSSF(50, 0.2, h111,0.02,30, "SSSF_zero_0.02_h111=0.2")
SSSF(50, 0.3, h111,0.02,30, "SSSF_zero_0.02_h111=0.3")
SSSF(50, 0.4, h111,0.02,30, "SSSF_zero_0.02_h111=0.4")

SSSF(50, 0.2, h001,0.02,30, "SSSF_zero_0.02_h001=0.2")
SSSF(50, 0.3, h001,0.02,30, "SSSF_zero_0.02_h001=0.3")
SSSF(50, 0.4, h001,0.02,30, "SSSF_zero_0.02_h001=0.4")
#
SSSF(50, 0.8, h110,0.02,30, "SSSF_zero_0.02_h110=0.8")
SSSF(50, 1.2, h110,0.02,30, "SSSF_zero_0.02_h110=1.2")
SSSF(50, 1.6, h110,0.02,30, "SSSF_zero_0.02_h110=1.6")

DSSF(0.04, 0, h111,-1/3, "DSSF_-0.33_detailed", 25, 0.04)
DSSF(0.02,0, h111, 0.046, "DSSF_0.046_detailed", 30, 0.02)



# #
#
# SSSF(25, 0, np.array([1,1,1]),-0.25,25, "SSSF_pi_-0.25_dumb")
# SSSF(25, 0, np.array([1,1,1]),-0.05,25, "SSSF_pi_-0.05_dumb")

#
# graphPhase("Files/phase_diagram.txt")


# findPhaseMag(0, 0.25, 35, 0, 3, 35, h111, 35, 1, "phase_test_111_kappa=1")
# findPhaseMag(0, 0.25, 35, 0, 3, 35, h001, 35, 1, "phase_test_001_kappa=1")
# findPhaseMag(0, 0.25, 35, 0, 12, 35, h110, 35, 1, "phase_test_110_kappa=1")
# graphMagPhase("phase_test_111_kappa=1", 0.25,3)
# graphMagPhase("phase_test_001_kappa=1", 0.25,3)
# graphMagPhase("phase_test_110_kappa=1", 0.25,12)

# graphMagPhase("phase_test_001", 0.25,3)
# graphMagPhase("phase_test_110", 0.25,12)