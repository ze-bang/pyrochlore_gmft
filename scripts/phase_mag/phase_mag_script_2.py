import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.phase_diagram import *
from core.misc_helper import *
from core.flux_stuff import *
from core.observables import *

# findPhaseMag111(-0.05, 0.05, 100, 0, 0.3, 100, h111, 30, 2, "phase_111_kappa=2_zoomed_in_more")
# findPhaseMag111_ex(-0.05, 0.05, 100, 0, 0.3, 100, h111, 17, 2, "phase_111_kappa=2_zoomed_in_more_ex")

Jpm = -0.03
# SSSF_line_pedantic(200, -2*Jpm, -2*Jpm, 1, 0, 0.4, 11, h111, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0", "hh2k", 0, 3, 3)
# SSSF_line_pedantic(200, -2*Jpm, -2*Jpm, 1, 0, 0.4, 11, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi", "hh2k", 0, 3, 3)

# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.5, 10, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi")


SSSF_plot_from_file('SSSF_hhl', 'hhl', Hr=4, Lr=4, cmap='viridis')
# SSSF_pedantic(100, 0.8, 0.8, 1, 0.0, h110, np.ones(4)*np.pi, 18, 'SSSF_hhl_Jpm=0.4', "hhl", Hr=4, Lr=4)
# SSSF_pedantic(100, 0.8, 0.8, 1, 0.0, h110, np.ones(4)*np.pi, 18, 'SSSF_hhknk_Jpm=0.4', "hhknk", Hr=4, Lr=4)
# SSSF(100, 0.011/0.063, 0.062/0.063, 1, 0.18, h110, np.ones(4)*np.pi, 30, 'Jxx_0.062_Jyy_0.063_Jzz_0.011_h110=0.2', "hhl")

# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 11, h111, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0", "hh2k", 0, 3, 3)
# SSSF_line_pedantic(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 11, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi", "hh2k", 0, 3, 3)


# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 3, h111, np.zeros(4),30, "Files/SSSF/Jpm=-0.03_0")
# SSSF_line(100, -2*Jpm, -2*Jpm, 1, 0, 0.4, 3, h111, np.ones(4)*np.pi, 30, "Files/SSSF/Jpm=-0.03_pi")


# DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h111, np.zeros(4), 30, "Files/DSSF/Jpm=-0.03_0")
# DSSF_line_pedantic(200,-2*Jpm, -2*Jpm, 1, 0, 0.4, 5, h111, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03_pi")
# DSSF(300, -2*Jpm, -2*Jpm, 1, 0, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.03/h110=0")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.1, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03/h110=0.1")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.2, h110, np.ones(4)*np.pi, 30, "Files/DSSF/Jpm=-0.03/h110=0.2")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.3, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.03/h110=0.3")
# DSSF(250, -2*Jpm, -2*Jpm, 1, 0.4, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.03/h110=0.4")
# DSSF(300, -2*Jpm, -2*Jpm, 1, 0.12, h110, np.array([0,0,np.pi,np.pi]), 30, "Files/DSSF/Jpm=-0.03/h110=0.12")


# findPhaseMag110(-0.3, 0.1, 40, 0, 0.5, 20, h110, 30, 2, "phase_110_kappa=2_Jpmpm=-0.2", Jpmpm=-0.2)
# SSSF_BZ(50, 0.08, 0.08, 1, 0, h110, np.zeros(4),15, "SSSF_BZ", 0)
# D = np.loadtxt("Szz.txt")
# D = D * 4 / (2*np.pi)*0.51550329757 * 18
# plt.imshow(D, origin='lower', extent=[-2.5,2.5,-2.5,2.5], aspect='auto')
# plt.colorbar()
# plt.savefig("Szz.pdf")