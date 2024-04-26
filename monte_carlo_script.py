import os

import monte_carlo

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from monte_carlo import *
from phase_diagram import *
from observables import *
# scan_all(h111, 2)
# scan_all(h110, 2)
# scan_all(h001, 2)


# downtetrahedron = np.array([[0,0,0],[0,1,0]])
# uptetrahedron = np.array([[0,0,0],[0,1,0]])
# ax = plt.axes(projection='3d')
# ax.set_axis_off()
# graphdownpyrochlore(downtetrahedron,ax)
# graphuppyrochlore(uptetrahedron,ax)
# # graphallgaugebond(downtetrahedron,ax)
# # temp = contract('ia,ar->ir',downtetrahedron,r)
# # hexa=np.array([temp[0]+monte_carlo.NN[3],temp[0]+monte_carlo.NN[2],temp[2]+monte_carlo.NN[0],temp[2]+monte_carlo.NN[3], temp[3]+monte_carlo.NN[2], temp[3]+monte_carlo.NN[0]])
# # plothexagon(hexa, ax)
# ax.view_init(elev=-36, azim=-45, roll=60)
# ax.set_aspect('equal')
# plt.show()
# plt.savefig('test.pdf')

# completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.zeros(4),'HanYan_100_Jpm_-0.3_0.1_h_0_1_0_flux_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.ones(4)*np.pi,'HanYan_100_Jpm_-0.3_0.1_h_0_1_pi_flux_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.array([np.pi, 0,0, np.pi]),'HanYan_100_Jpm_-0.3_0.1_h_0_1_0pipi0_ex')
# completeSpan_ex(-0.3,0.1,100,0,1,100,h001,17,2,np.array([0,np.pi,np.pi,0]),'HanYan_100_Jpm_-0.3_0.1_h_0_1_0_pi00pi_ex')

# Jpm = -0.289

# SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 1.25, 1.75, 25, 30, "Files/Jpm=-0.289_h110=0.1_HHKnK_pi_flux")
# SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 1.25, 1.75, 25, 30, "Files/Jpm=-0.289_h110=0.2_HHKnK_pi_flux")
# SSSF_HHKnK_L_integrated(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.ones(4)*np.pi, 1.25, 1.75, 25, 30, "Files/Jpm=-0.289_h110=0.3_HHKnK_pi_flux")

# SSSF_HHL_KK_integrated(100, -2*Jpm, -2*Jpm, 1, 0.05, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, 'Jpm=-0.289_h110=0.05_L_integrated')
# SSSF_HHL_KK_integrated(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, 'Jpm=-0.289_h110=0.1_L_integrated')
# SSSF_HHL_KK_integrated(100, -2*Jpm, -2*Jpm, 1, 0.15, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, 'Jpm=-0.289_h110=0.15_L_integrated')

# DSSF(0.01, 0.062/0.063, 0.011/0.063, 1, 0, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jxx=0.011_Jyy=0.063_Jzz=0.062_h110=0")
# DSSF(0.01, 0.062/0.063, 0.011/0.063, 1, 0.05, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jxx=0.011_Jyy=0.063_Jzz=0.062_h110=0.05")
# DSSF(0.01, 0.062/0.063, 0.011/0.063, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jxx=0.011_Jyy=0.063_Jzz=0.062_h110=0.1")

# SSSF_HHL_KK_integrated(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.05, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.063_Jyy=0.062_Jzz=0.011_h110=0.05_HHL_KnK_integrated")
# SSSF_HHL_KK_integrated(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.15, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.063_Jyy=0.062_Jzz=0.011_h110=0.15_HHL_KnK_integrated")


# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h001, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h001=0.1_HK0_pi_flux", "hk0", 0)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h001, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h001=0.2_HK0_pi_flux", "hk0", 0)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h001, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h001=0.3_HK0_pi_flux", "hk0", 0)

findPhaseMag110(-0.5, 0.1, 250, 0, 2.2, 200, h110, 30, 2, "phase_110_kappa=2")
findPhaseMag111(-0.5, 0.1, 250, 0, 0.7, 200, h111, 30, 2, "phase_111_kappa=2")
findPhaseMag001(-0.5, 0.1, 250, 0, 0.5, 200, h001, 30, 2, "phase_100_kappa=2")
