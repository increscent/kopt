#!/usr/local/bin/python3.7

import math
import random
import signal
import sys
import time


from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtWidgets import *
	from PyQt5.QtGui import *
	from PyQt5.QtCore import *
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtGui import *
	from PyQt4.QtCore import *
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


#TODO: Error checking on txt boxes
#TODO: Color strings


# Import in the code with the actual implementation
from TSPSolver import *
from TSPClasses import *


class Proj5GUI():

    def __init__(self, seed, size):
        self._scenario = None
        self.solver = TSPSolver(None)
        self.seed = seed
        self.size = size
        self.maxTime = 60
        self.diff = 'Hard'
        self.data_range = {'x': [-1.5, 1.5], 'y': [-1, 1]}

        self.generateNetwork()
        self.solve()
       
    def newPoints(self):		
        seed = int(self.seed)
        random.seed(seed)

        ptlist = []
        RANGE = self.data_range
        xr = self.data_range['x']
        yr = self.data_range['y']
        npoints = int(self.size)
        while len(ptlist) < npoints:
            x = random.uniform(0.0,1.0)
            y = random.uniform(0.0,1.0)
            if True:
                xval = xr[0] + (xr[1]-xr[0])*x
                yval = yr[0] + (yr[1]-yr[0])*y
                ptlist.append( QPointF(xval,yval) )
        return ptlist

    def generateNetwork(self):
        points = self.newPoints() # uses current rand seed
        rand_seed = self.seed
        self._scenario = Scenario( city_locations=points, difficulty=self.diff, rand_seed=rand_seed )

    def solve(self):
        self.solver.setupWithScenario(self._scenario)

        max_time = self.maxTime

        solve_func = 'self.solver.branchAndBound'
        results = eval(solve_func)(time_allowance=max_time)

        print((self.size, self.seed, results['time'], results['cost'],\
            results['max'], results['count'], results['total'], results['pruned']))

    ALGORITHMS = [ \
        ('Default                            ','defaultRandomTour'), \
        ('Greedy','greedy'), \
        ('Branch and Bound','branchAndBound'), \
        ('Fancy','fancy') \
    ]

if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal.SIG_DFL)
	
	w = Proj5GUI(int(sys.argv[2]), int(sys.argv[1]))
