import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# import numpy as np
from variation_flux import *

plot_MFE_flux_110_alt(0, 0, -0.008, -0.008, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_mid_0.3_n1=0_n2=0_alt")
plot_MFE_flux_110_alt(0, 0, 0.1, 0.1, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_pi_0.3_n1=0_n2=0_alt")
plot_MFE_flux_110_alt(0, 0, -0.05, -0.05, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_zero_0.3_n1=0_n2=0_alt")

plot_MFE_flux_110_alt(1, 0, -0.008, -0.008, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_mid_0.3_n1=1_n2=0_alt")
plot_MFE_flux_110_alt(1, 0, 0.1, 0.1, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_pi_0.3_n1=1_n2=0_alt")
plot_MFE_flux_110_alt(1, 0, -0.05, -0.05, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_zero_0.3_n1=1_n2=0_alt")

plot_MFE_flux_110_alt(0, 1, -0.008, -0.008, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_mid_0.3_n1=0_n2=1_alt")
plot_MFE_flux_110_alt(0, 1, 0.1, 0.1, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_pi_0.3_n1=0_n2=1_alt")
plot_MFE_flux_110_alt(0, 1, -0.05, -0.05, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_zero_0.3_n1=0_n2=1_alt")

plot_MFE_flux_110_alt(1, 1, -0.008, -0.008, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_mid_0.3_n1=1_n2=1_alt")
plot_MFE_flux_110_alt(1, 1, 0.1, 0.1, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_pi_0.3_n1=1_n2=1_alt")
plot_MFE_flux_110_alt(1, 1, -0.05, -0.05, 1, 0.3, h110, 2, 40, 101, "h110_flux_plane_zero_0.3_n1=1_n2=1_alt")