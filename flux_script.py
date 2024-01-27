import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *
from variation_flux import *


plot_MFE_flux_110(1, 0, -0.001, -0.001, 1, 0.2, h110, 2, 30, 101, "h110_flux_plane_mid_0.2_n1=1_n2=0")
plot_MFE_flux_110(1, 0, 0.1, 0.1, 1, 0.2, h110, 2, 30, 101, "h110_flux_plane_pi_0.2_n1=1_n2=0")
plot_MFE_flux_110(1, 0, -0.08, -0.08, 1, 0.2, h110, 2, 30, 101, "h110_flux_plane_zero_0.2_n1=1_n2=0")


# plot_MFE_flux_110(-0.001, -0.001, 1, 0.25, h110, 2, 40, 101, "h110_flux_plane_mid_0.25")
# plot_MFE_flux_110(0.1, 0.1, 1, 0.25, h110, 2, 40, 101, "h110_flux_plane_pi_0.25")
# plot_MFE_flux_110(-0.08, -0.08, 1, 0.25, h110, 2, 40, 101, "h110_flux_plane_zero_0.25")

# plot_MFE_flux_111(1, -0.001, -0.001, 1, 0.3, h111, 2, 30, 101, "h111_flux_plane_mid_0.3_n1=1")
# plot_MFE_flux_111(1, 0.1, 0.1, 1, 0.3, h111, 2, 30, 101, "h111_flux_plane_pi_0.3_n1=1")
# plot_MFE_flux_111(1, -0.08, -0.08, 1, 0.3, h111, 2, 30 , 101, "h111_flux_plane_zero_0.3_n1=1")


# A = flux_converge_line(-0.4, 0.4, 40, 0.3, h111, 2, 26, 4, "h111_flux_line")
# B = flux_converge_line(-0.4, 0.4, 40, 0.3, h110, 2, 26, 4, "h111_flux_line")
# np.savetxt("h111_flux_line.txt", A)
# np.savetxt("h110_flux_line.txt", B)
