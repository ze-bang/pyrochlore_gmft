import os
import numpy as np
# os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.phase_diagram import *
from core.observables import *
import sys


# findPhaseMag_separate(Jpm_length_start, Jpm_length_end, nK, h_length_start, h_length_end, nK, h_dir, flux, 30, 2, filename, Jpmpm=Jpmpm, FF=FF)
