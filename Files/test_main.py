import matplotlib.pyplot as plt
from misc_helper import *
import pyrochlore_dispersion as py0
import pyrochlore_dispersion_pi as pypi


def graphdispersion(JP,h, n, kappa, rho, graphres, BZres):
    py0s = pypi.piFluxSolver(JP,eta=kappa, kappa=rho, graphres=graphres, BZres=BZres, h=h, n=n)
    py0s.findminLam()
    print(py0s.minLams)
    py0s.findLambda()
    print(py0s.lams)
    print(py0s.condensed())
    py0s.qvec()
    print(py0s.q)
    py0s.graph(True)

graphdispersion(-0.1, 0.2, h001, 1, 2, 20, 20)
