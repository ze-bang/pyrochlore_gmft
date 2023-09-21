import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *

SSSF(100, 0.8, h1b10,hb110, -0.1, 35, "SSSF_pi_-0.1_h1b10=0.8")
SSSF(100, 1.0, h1b10,hb110, -0.1, 35, "SSSF_pi_-0.1_h1b10=1.0")
SSSF(100, 1.2, h1b10,hb110, -0.1, 35, "SSSF_pi_-0.1_h1b10=1.2")

SSSF(100, 0.8, h110,hb110, -0.1, 35, "SSSF_pi_-0.1_h110=0.8")
SSSF(100, 1.0, h110,hb110, -0.1, 35, "SSSF_pi_-0.1_h110=1.0")
SSSF(100, 1.2, h110,hb110, -0.1, 35, "SSSF_pi_-0.1_h110=1.2")