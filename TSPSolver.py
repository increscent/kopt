#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import math
import numpy as np
from TSPClasses import *
import heapq
import itertools
from pqueue import *



class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None

    def setupWithScenario( self, scenario ):
        self._scenario = scenario


    ''' <summary>
            This is the entry point for the default solver
            which just finds a valid random tour.  Note this could be used to find your
            initial BSSF.
            </summary>
            <returns>results dictionary for GUI that contains three ints: cost of solution, 
            time spent to find solution, number of permutations tried during search, the 
            solution found, and three null values for fields not used for this 
            algorithm</returns> 
    '''
    
    def defaultRandomTour( self, time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        n = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation( n )
            route = []
            # Now build the route using the random permutation
            for i in range( n ):
                route.append( cities[ perm[i] ] )
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results


    ''' <summary>
            This is the entry point for the greedy solver, which you must implement for 
            the group project (but it is probably a good idea to just do it for the branch-and
            bound project as a way to get your feet wet).  Note this could be used to find your
            initial BSSF.
            </summary>
            <returns>results dictionary for GUI that contains three ints: cost of best solution, 
            time spent to find best solution, total number of solutions found, the best
            solution found, and three null values for fields not used for this 
            algorithm</returns> 
    '''

    def greedy( self,time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()[:]
        foundTour = False
        bssf = None

        path = []
        path.append(cities[0])
        curCity = cities[0]
        del cities[0]

        start_time = time.time()
        while len(cities) > 0:
            minDistance = curCity.costTo(cities[0])
            minIndex = 0

            for i in range(1, len(cities)):
                if curCity.costTo(cities[i]) < minDistance:
                    minDistance = curCity.costTo(cities[i])
                    minIndex = i

            if minDistance == math.inf:
                # no path found
                break

            curCity = cities[minIndex]
            path.append(curCity)
            del cities[minIndex]

            # check for path back to start city
            if len(cities) == 0 and curCity.costTo(path[0]) < math.inf:
                foundTour = True
            
        end_time = time.time()

        bssf = TSPSolution(path) if foundTour else None

        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = 0
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results
    
    
    
    ''' <summary>
            This is the entry point for the branch-and-bound algorithm that you will implement
            </summary>
            <returns>results dictionary for GUI that contains three ints: cost of best solution, 
            time spent to find best solution, total number solutions found during search (does
            not include the initial BSSF), the best solution found, and three more ints: 
            max queue size, total number of states created, and number of pruned states.</returns> 
    '''
            
    def branchAndBound( self, time_allowance=60.0 ):
        start_time = time.time()

        # initialization
        results = {}
        cities = self._scenario.getCities()
        n = len(cities)
        count = 0
        maxSize = 0
        totalNodes = 0
        pruned = 0
        bssf = math.inf
        bestPath = None
        # use a heap implementation of a priority queue
        pqueue = PqueueHeap() 

        # build initial cost matrix -- O(n^2)
        costm = [[math.inf for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                costm[i][j] = cities[i].costTo(cities[j])

        lowerBound = reduceMatrix(costm)

        # get initial bssf (random)
        # average case: O(n)
        # worst case: O(n!)
        randomResult = self.defaultRandomTour()
        initialBssf = randomResult['cost']
        bssf = initialBssf

        # start with an initial branch
        branch = Branch(0, lowerBound, costm, 0, [0])

        # maximum number of branches O(n^(n^3))
        while branch != None:
            # stop if time is up
            if time.time() - start_time > time_allowance:
                break

            # prune it
            if branch.lowerBound >= bssf:
                branch = pqueue.deleteMin()
                pruned += 1
                continue
            
            # loop through possible child nodes -- O(n^3)
            # there are n possible child nodes and reducing the matrix takes O(n^3) time
            for j in range(n):
                if j == branch.city or branch.matrix[branch.city][j] == math.inf:
                    continue

                totalNodes += 1

                # copy the cost matrix
                newCostm = list(map(lambda row: row[:], branch.matrix))
                newLowerBound = branch.lowerBound + newCostm[branch.city][j]

                # it is impossible to leave this city again
                newCostm[branch.city] = [math.inf for _ in range(n)]

                # it is also impossible to enter the new city again
                for i in range(n):
                    newCostm[i][j] = math.inf

                # copy the path and add this city
                newPath = branch.path[:]
                newPath.append(j)

                # check if the path is complete
                if len(newPath) == n:
                    # done (add the cost of returning to the start node)
                    totalCost = newLowerBound + newCostm[j][newPath[0]]

                    # check if this is the best solution so far
                    if totalCost < bssf:
                        bssf = totalCost
                        bestPath = newPath
                        count += 1

                else:
                    # don't return to the first element
                    newCostm[j][newPath[0]] = math.inf

                    # reduce the matrix
                    newLowerBound += reduceMatrix(newCostm)

                    # priority is a ratio of cost to level
                    priority = newLowerBound / len(newPath)

                    # prune that
                    if newLowerBound >= bssf:
                        pruned += 1
                        continue

                    pqueue.insert(Branch(priority, newLowerBound, newCostm, j, newPath), priority)
                    
                    # check for max priority queue size
                    if pqueue.fill > maxSize:
                        maxSize = pqueue.fill

            # get next branch from priority queue
            branch = pqueue.deleteMin()

        end_time = time.time()

        # compute route from city indeces if a better solution was found than the initial one
        if bssf < initialBssf:
            route = list(map(lambda x: cities[x], bestPath)) if bssf < math.inf else []

        results['cost'] = bssf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = TSPSolution(route) if bssf < initialBssf else randomResult['soln']
        results['max'] = maxSize
        results['total'] = totalNodes
        results['pruned'] = pruned + pqueue.fill

        return results



    ''' <summary>
            This is the entry point for the algorithm you'll write for your group project.
            </summary>
            <returns>results dictionary for GUI that contains three ints: cost of best solution, 
            time spent to find best solution, total number of solutions found during search, the 
            best solution found.  You may use the other three field however you like.
            algorithm</returns> 
    '''
            
    def fancy( self,time_allowance=60.0 ):
        pass
            

class Branch:
    def __init__(self, priority, lowerBound, costMatrix, city, path):
        self.priority = priority
        self.lowerBound = lowerBound
        self.matrix = costMatrix
        self.city = city
        self.path = path

# O(n^2)
def reduceMatrix(m):
    n = len(m)
    cost = 0

    # reduce each row -- O(n^2)
    for i in range(n):
        minVal = min(m[i])
        
        if minVal < math.inf:
            cost += minVal
            m[i] = list(map(lambda x: x - minVal, m[i]))

    # reduce each column -- O(n^2)
    for j in range(n):
        minVal = min(list(map(lambda row: row[j], m)))

        if minVal < math.inf:
            cost += minVal
            for i in range(n):
                m[i][j] -= minVal

    return cost

