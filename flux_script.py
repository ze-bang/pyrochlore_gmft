import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from variation_flux import *


plot_MFE_flux_111(0, -0.008, -0.008, 1, 0.3, h111, 2, 40, 101, "h111_flux_plane_mid_0.3_n1=0")
plot_MFE_flux_111(0, 0.1, 0.1, 1, 0.3, h111, 2, 40, 101, "h111_flux_plane_pi_0.3_n1=0_n2=0")
plot_MFE_flux_111(0, -0.05, -0.05, 1, 0.3, h111, 2, 40, 101, "h111_flux_plane_zero_0.3_n1=0_n2=0")

plot_MFE_flux_111(1, -0.008, -0.008, 1, 0.3, h111, 2, 40, 101, "h111_flux_plane_mid_0.3_n1=1")
plot_MFE_flux_111(1, 0.1, 0.1, 1, 0.3, h111, 2, 40, 101, "h111_flux_plane_pi_0.3_n1=1")
plot_MFE_flux_111(1, -0.05, -0.05, 1, 0.3, h111, 2, 40, 101, "h111_flux_plane_zero_0.3_n1=1")
