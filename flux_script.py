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


plot_MFE_flux_111(0, 0, -0.001, -0.001, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_mid_0.2_n1=0_n2=0")
plot_MFE_flux_111(0, 0, 0.1, 0.1, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_pi_0.2_n1=0_n2=0")
plot_MFE_flux_111(0, 0, -0.08, -0.08, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_zero_0.2_n1=0_n2=0")

plot_MFE_flux_111(1, 0, -0.001, -0.001, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_mid_0.2_n1=1_n2=0")
plot_MFE_flux_111(1, 0, 0.1, 0.1, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_pi_0.2_n1=1_n2=0")
plot_MFE_flux_111(1, 0, -0.08, -0.08, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_zero_0.2_n1=1_n2=0")

plot_MFE_flux_111(0, 1, -0.001, -0.001, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_mid_0.2_n1=0_n2=1")
plot_MFE_flux_111(0, 1, 0.1, 0.1, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_pi_0.2_n1=0_n2=1")
plot_MFE_flux_111(0, 1, -0.08, -0.08, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_zero_0.2_n1=0_n2=1")

plot_MFE_flux_111(1, 1, -0.001, -0.001, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_mid_0.2_n1=1_n2=1")
plot_MFE_flux_111(1, 1, 0.1, 0.1, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_pi_0.2_n1=1_n2=1")
plot_MFE_flux_111(1, 1, -0.08, -0.08, 1, 0.2, h111, 2, 30, 101, "h111_flux_plane_zero_0.2_n1=1_n2=1")
