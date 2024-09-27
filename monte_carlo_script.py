import os
import sys
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# import monte_carlo
# from monte_carlo import *
from phase_diagram import *
# from observables import *
# scan_all(h111, 2)
# scan_all(h110, 2)
# scan_all(h001, 2)


# downtetrahedron = np.array([[0,0,0],[0,1,0],[1,0,0],[0,0,1],[0,2,0],[2,0,0],[-1,1,1],[1,-1,1], [1,1,-1], [0,1,1], [1,1,0], [1,1,1]])
# uptetrahedron =  np.array([[0,1,1], [1,0,1], [1,1,0], [1,1,1]])
# downtetrahedron = np.array([[0,0,0]])
# uptetrahedron =  np.array([[0,0,0]])
# ax = plt.axes(projection='3d')
# ax.set_axis_off()

# graphdownpyrochlore(downtetrahedron,ax)
# graphuppyrochlore(uptetrahedron,ax)
# graphdownpyrochlore_blank(downtetrahedron,ax)
# graphuppyrochlore_blank(uptetrahedron,ax)
# points = np.array([[0,0,0],[]])
# temp = contract('ia,ar->ir',downtetrahedron,r)
# # ANN = -NN
# hexa320=np.array([temp[0]+NN[3],temp[0]+NN[2],temp[2]+NN[0],temp[2]+NN[3], temp[3]+NN[2], temp[3]+NN[0]])
# hexa321=np.array([temp[1]+NN[3],temp[1]+NN[2],temp[2]+NN[1],temp[2]+NN[3], temp[3]+NN[2], temp[3]+NN[1]])
# hexa301=np.array([temp[1]+NN[3],temp[1]+NN[0],temp[0]+NN[1],temp[0]+NN[3], temp[3]+NN[0], temp[3]+NN[1]])
# hexa201=np.array([temp[1]+NN[2],temp[1]+NN[0],temp[0]+NN[1],temp[0]+NN[2], temp[2]+NN[0], temp[2]+NN[1]])

# plothexagon(hexa320, ax, 'r')
# plothexagon(hexa321, ax,'r')
# # plothexagon(hexa301, ax, 'r')
# plothexagon(hexa201, ax,'r')
#
# ax.plot([0,0.75],[0,0.75],[0,0], color='black', linewidth=1.2)
# ax.plot([0,0],[0,0.75],[0,0.75], color='black', linewidth=1.2)
# ax.plot([0,0.75],[0,0],[0,0.75], color='black', linewidth=1.2)
# ax.plot([0,0.75],[0,0],[0,0.75], color='black', linewidth=1.2)
# ax.plot([0.75,0.75],[0.75,0],[0,0.75], color='black', linewidth=1.2)
# ax.plot([0,0.75],[0.75,0],[0.75,0.75], color='black', linewidth=1.2)
# ax.plot([0,0.75],[0.75,0.75],[0.75,0], color='black', linewidth=1.2)
# ax.plot([0,1],[0,0],[0,0], color='black', linewidth=0.3)
# ax.plot([0,0],[0,1],[0,0], color='black', linewidth=0.3)
# ax.plot([0,0],[0,0],[0,1], color='black', linewidth=0.3)
# ax.plot([1,1],[1,0],[0,0], color='black', linewidth=0.3)
# ax.plot([1,1],[0,0],[1,0], color='black', linewidth=0.3)
# ax.plot([1,1],[0,1],[1,1], color='black', linewidth=0.3)
# ax.plot([1,1],[1,1],[1,0], color='black', linewidth=0.3)
# ax.plot([0,1],[1,1],[1,1], color='black', linewidth=0.3)
# ax.plot([0,1],[1,1],[0,0], color='black', linewidth=0.3)
# ax.plot([0,0],[1,1],[1,0], color='black', linewidth=0.3)
# ax.plot([0,0],[1,0],[1,1], color='black', linewidth=0.3)
# ax.plot([1,0],[0,0],[1,1], color='black', linewidth=0.3)
# graphallgaugebond(downtetrahedron,ax)

# ax.view_init(elev=0, azim=84, roll=45)
# ax.set_aspect('equal')
# plt.show()
# plt.savefig('./Misc/0pp0.pdf')

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

# DSSF(0.01, 0.062/0.063, 0.011/0.063, 1, 0.15, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jxx=0.011_Jyy=0.063_Jzz=0.062_h110=0.15")
# DSSF(0.01, 0.062/0.063, 0.011/0.063, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jxx=0.011_Jyy=0.063_Jzz=0.062_h110=0.1")

# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/Jpm=-0.289_h110=0.1_HHKnK_pi_flux_1.5_0.3", "hkk", 1.5, 1.5, 0.3)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 30, "Files/Jpm=-0.289_h110=0.2_HHKnK_pi_flux_1.5_0.3", "hkk", 1.5, 1.5, 0.3)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.ones(4)*np.pi, 30, "Files/Jpm=-0.289_h110=0.3_HHKnK_pi_flux_1.5_0.3", "hkk", 1.5, 1.5, 0.3)
# SSSF_HHL_KK_integrated(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.05, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.063_Jyy=0.062_Jzz=0.011_h110=0.05_HHL_KnK_integrated")
# SSSF_HHL_KK_integrated(100, 0.063/0.063, 0.062/0.063, 0.011/0.063, 0.15, h110, np.ones(4)*np.pi, -0.3, 0.3, 51, 30, "Files/SSSF/CZO_gaulin/Jxx=0.063_Jyy=0.062_Jzz=0.011_h110=0.15_HHL_KnK_integrated")


# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.1, h001, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h001=0.1_HK0_pi_flux", "hk0", 0)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.2, h001, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h001=0.2_HK0_pi_flux", "hk0", 0)
# SSSF(100, -2*Jpm, -2*Jpm, 1, 0.3, h001, np.ones(4)*np.pi, 30, "Files/Jpm=-0.03_h001=0.3_HK0_pi_flux", "hk0", 0)

# findPhaseMag110(-0.5, 0.1, 300, 0, 2.2, 200, h110, 30, 2, "phase_110_kappa=2")
# findPhaseMag111(-0.5, 0.1, 300, 0, 0.7, 200, h111, 30, 2, "phase_111_kappa=2")
# findPhaseMag111(-0.5, 0.1, 300, 0, 0.5, 200, h001, 30, 2, "phase_001_kappa=2")
# TwoSpinonDOS_111_a(50, 20, "h=0.3_2SPINONS")

#findXYZPhase_separate(-0.5, 1, -0.5, 1, 40, 30, 2, np.zeros(4), "phase_XYZ_0_field_0_flux", 0)

# Jpm, Jpmax, Jpm1, Jpm1max, nK = sys.argv[1:]
Jpm, Jpmax, Jpm1, Jpm1max, nK = -1, 0, -1, 0, 40
findXYZPhase_separate(Jpm, Jpmax, Jpm1, Jpm1max, nK, 30, 2, np.zeros(4), "phase_XYZ_0_field_0_flux_-1_0_-1_0", 0, symmetrized=False)
