import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi
import numpy as np
import matplotlib.pyplot as plt
from spinon_con import *
from phase_diagram import *
from misc_helper import *


TWOSPINCON(80, 0, h111, -1/3, 25, "TSC_-0.33_h111=0")
TWOSPINCON(80, 0.2, h111, -1/3, 25, "TSC_-0.33_h111=0.2")
TWOSPINCON(80, 1, h111, -1/3, 25, "TSC_-0.33_h111=1.0")

TWOSPINCON(80, 0.2, h001, -1/3, 25, "TSC_-0.33_h001=0.2")
TWOSPINCON(80, 1, h001, -1/3, 25, "TSC_-0.33_h001=1.0")

TWOSPINCON(80, 0.2, h110, -1/3, 25, "TSC_-0.33_h110=0.2")
TWOSPINCON(80, 1, h110, -1/3, 25, "TSC_-0.33_h110=1.0")