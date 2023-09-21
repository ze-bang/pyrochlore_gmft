import pyrochlore_dispersion
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import pyrochlore_dispersion_pi as pygang
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
import math
import time
import sys
from phase_diagram import *
from numba import jit
from misc_helper import *
import netCDF4 as nc


findPhaseMag(-0.5, 0.1, 300, 0, 1, 150, h1b10, 25, 2, "phase_1b10_kappa=2_complete")

findPhaseMag_pi_zero(-0.5, 0, 300, 0, 1, 150, h111, 25, 2, "phase_111_kappa=2_complete")
findPhaseMag_pi_zero(-0.5, 0, 300, 0, 1, 150, h001, 25, 2, "phase_001_kappa=2_complete")
findPhaseMag_pi_zero(-0.5, 0, 300, 0, 4, 150, h110, 25, 2, "phase_110_kappa=2_complete")