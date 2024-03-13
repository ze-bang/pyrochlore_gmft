from monte_carlo import *
import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

scan_all(h111, 2)
scan_all(h110, 2)
scan_all(h001, 2)