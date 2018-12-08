import TSPSolver
import TSPClasses

class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y
        

t = TSPSolver.TSPSolver(None)

s = TSPClasses.Scenario([Point(0, 0), Point(0.5, 1), Point(2, 1), Point(3, 0)], "Easy", 1) 

t.setupWithScenario(s)

t.branchAndBound()
