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


flux_converge_line(-0.05, 0.05, 50, 0.1, h110, 2, 35, 3, 'flux_converge_h110')


# plot_MFE_flux_110(-0.001, -0.001, 1, 0.25, h110, 2, 40, 101, "h110_flux_plane_mid_0.25")
# plot_MFE_flux_110(0.1, 0.1, 1, 0.25, h110, 2, 40, 101, "h110_flux_plane_pi_0.25")
# plot_MFE_flux_110(-0.08, -0.08, 1, 0.25, h110, 2, 40, 101, "h110_flux_plane_zero_0.25")

# plot_MFE_flux_111(-0.001, -0.001, 1, 0.3, h111, 2, 26, 101, "h111_flux_plane_mid")
# plot_MFE_flux_111(0.1, 0.1, 1, 0.3, h111, 2, 26, 101, "h111_flux_plane_pi")
# plot_MFE_flux_111(-0.08, -0.08, 1, 0.3, h111, 2, 26, 101, "h111_flux_plane_zero")


# A = flux_converge_line(-0.4, 0.4, 40, 0.3, h111, 2, 26, 4, "h111_flux_line")
# B = flux_converge_line(-0.4, 0.4, 40, 0.3, h110, 2, 26, 4, "h111_flux_line")
# np.savetxt("h111_flux_line.txt", A)
# np.savetxt("h110_flux_line.txt", B)
