import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from phase_diagram import *
import time



start = time.time()
findPhaseMag110(-0.5, 0.1, 1, 0, 1, 100, h111, 25, 2, "FF_phase_111_kappa=2")
end = time.time()
print(end-start)
