#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import copy
import random


class Context:
    def __init__(self, currentCity, cities, costMatrix, depth, lowerBound, route):
        self.currentCity = currentCity
        self.cityIndexes = cities
        self.costMatrix = costMatrix
        self.depth = depth
        self.lowerBound = lowerBound
        self.sort = lowerBound / (depth + 1)  # or lowerBound - constant*depth
        self.route = route

    def __lt__(self, other):
        if self.sort < other.sort:
            return True
        return False


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None
        self.bssf = float('Inf')
        self.solution = None
        self.route = None
        self.count = 0
        self.cities = None
        self.startTime = 0

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour
        </summary>
        <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (
not counting initial BSSF estimate)</returns> '''

    def defaultRandomTour(self, start_time, time_allowance=60.0):

        results = {}

        start_time = time.time()

        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        while not foundTour:
            # create a random permutation
            perm = np.random.permutation(ncities)

            # for i in range( ncities ):
            # swap = i
            # while swap == i:
            # swap = np.random.randint(ncities)
            # temp = perm[i]
            # perm[i] = perm[swap]
            # perm[swap] = temp

            route = []

            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])

            bssf = TSPSolution(route)
            # bssf_cost = bssf.cost()
            # count++;
            count += 1

            # if costOfBssf() < float('inf'):
            if bssf.costOfRoute() < np.inf:
                # Found a valid route
                foundTour = True
        # } while (costOfBssf() == double.PositiveInfinity);                // until a valid route is found
        # timer.Stop();

        results['cost'] = bssf.costOfRoute()  # costOfBssf().ToString();                          // load results array
        results['time'] = time.time() - start_time
        results['count'] = count
        results['soln'] = bssf

        # return results;
        return results

    def greedyLowerBound(self):

        startTime = time.time()
        results = {}
        startCity = 0
        bssf = None
        while True:
            cities = copy.deepcopy(self._scenario.getCities())
            currentCity = cities.pop(startCity)
            route = [currentCity]
            while len(cities) != 0:
                currentCity = self.shortest(currentCity, cities)
                route.append(currentCity)
            bssf = TSPSolution(route)
            if bssf.costOfRoute() != float('Inf'):
                break
            startCity += 1

        results['cost'] = bssf.costOfRoute()
        results['time'] = time.time() - startTime
        results['count'] = 1
        results['soln'] = bssf
        # return bssf.costOfRoute()
        return results

    def setInfinities(self, theMatrix, current, end):
        for colNum in range(theMatrix.shape[1]):
            theMatrix[current, colNum] = float("Inf")
        for rowNum in range(theMatrix.shape[0]):
            theMatrix[rowNum, end] = float("Inf")
        return theMatrix

    def reduce_matrix(self, theMatrix):
        summation = 0
        for rowNum in range(theMatrix.shape[0]):
            theRow = theMatrix[rowNum, :]
            smallest = np.amin(theRow)
            if smallest > 0 and smallest != float("Inf"):
                summation += smallest
                for colNum in range(len(theRow)):
                    theMatrix[rowNum, colNum] = theMatrix[rowNum, colNum] - smallest

        for colNum in range(theMatrix.shape[1]):
            theCol = theMatrix[:, colNum]
            smallest = np.amin(theCol)
            if smallest > 0 and smallest != float("Inf"):
                summation += smallest
                for rowNum in range(len(theCol)):
                    theMatrix[rowNum, colNum] = theMatrix[rowNum, colNum] - smallest

        return summation

    def convertRoute(self, route):
        route2 = []
        for index in route:
            route2.append(self.cities[index])
        return route2

    def find_best(self, heap):

        maxStates = 0
        totalStates = 0
        pruned = 0
        while True:
            if len(heap) > maxStates:
                maxStates = len(heap)
            # We done if the heapq is empty
            if len(heap) == 0:
                break

            if time.time() - self.startTime > 600:
                break

            # Get next context
            contextCopy = heapq.heappop(heap)
            # Reduce
            contextCopy.lowerBound = self.reduce_matrix(contextCopy.costMatrix) + contextCopy.lowerBound

            bssf_cost = np.math.inf
            try:
                # print("bssf: ", self.bssf['cost'])
                bssf_cost = self.bssf['cost']
            except TypeError:
                # print("bssf whole object: ", self.bssf)
                bssf_cost = self.bssf

            # Prune
            if contextCopy.lowerBound >= bssf_cost:
                pruned += 1
                continue  # Don't expand children and get the next thing on the queue

            # Found a solution
            if len(contextCopy.cityIndexes) == 0:
                if contextCopy.costMatrix[contextCopy.route[-1], 0] == float("Inf"):
                    continue
                theRoute = self.convertRoute(contextCopy.route)
                self.solution = TSPSolution(theRoute)
                self.bssf = self.solution.costOfRoute()
                self.route = contextCopy.route
                self.count += 1
                continue

            # Expand
            for index in contextCopy.cityIndexes:
                newLower = contextCopy.lowerBound + contextCopy.costMatrix[contextCopy.currentCity, index]
                newMatrix = self.setInfinities(copy.deepcopy(contextCopy.costMatrix),
                                               copy.deepcopy(contextCopy.currentCity), index)
                newCities = copy.deepcopy(contextCopy.cityIndexes)
                newCurrent = index
                newCities.remove(index)
                newContext = Context(newCurrent, newCities, newMatrix, (len(contextCopy.route) + 1), newLower,
                                     contextCopy.route + [index])
                heapq.heappush(heap, newContext)
                totalStates += 1
                if time.time() - self.startTime > 60:
                    break
        print("Max Stored", maxStates)
        print("BSSF updates", self.count)
        print("Total States Created", totalStates)
        print("Pruned", pruned)

        return self.solution

    def branchAndBound(self, start_time, time_allowance=60.0):
        # choose the best lowerBound
        self.bssf = self.greedyLowerBound()
        # lowerBoundR = self.defaultRandomTour()
        # self.bssf = min([lowerBoundG, lowerBoundR])

        # print(lowerBoundG)
        # print(lowerBoundR)
        print(self.bssf)

        results = {}
        start_time = time.time()
        self.startTime = start_time
        cities = copy.deepcopy(self._scenario.getCities())
        # build cost matrix
        self.cities = cities
        theMatrix = []
        for i in range(len(cities)):
            row = []
            for j in range(len(cities)):
                row.append(cities[i].costTo(cities[j]))
            theMatrix.append(row)
        costMatrix = np.array(theMatrix)
        cityIndexes = []
        for i in range(len(cities)):
            cityIndexes.append(i)
        startCity = cityIndexes.pop(0)
        depth = 0
        lowerBound = 0
        route = [0]
        context = Context(startCity, cityIndexes, costMatrix, depth, lowerBound, route)
        heap = []
        heapq.heappush(heap, context)  # It will sort based on the depth and lowerBound given
        best = self.find_best(heap)

        print('\n')
        results['cost'] = best.costOfRoute()
        final_time = time.time() - start_time
        results['time'] = final_time
        print("Time: ", final_time)
        results['count'] = self.count
        results['soln'] = best
        return results

    def shortest(self, currentCity, cities):
        shortest = float("Inf")
        index = 0
        for i in range(len(cities)):
            cost = currentCity.costTo(cities[i])
            if cost < shortest:
                shortest = cost
                index = i
        return cities.pop(index)

    def greedy(self, start_time, time_allowance=60.0):
        startTime = time.time()
        results = {}
        startCity = 0
        bssf = None
        while True:
            cities = copy.deepcopy(self._scenario.getCities())
            currentCity = cities.pop(startCity)
            route = [currentCity]
            while len(cities) != 0:
                currentCity = self.shortest(currentCity, cities)
                route.append(currentCity)
            bssf = TSPSolution(route)
            if bssf.costOfRoute() != float('Inf'):
                break
            startCity += 1

        results['cost'] = bssf.costOfRoute()
        results['time'] = time.time() - startTime
        results['count'] = 1
        results['soln'] = bssf
        return results

    def two_opt(self, start_time, time_allowance=60.0):
        best_cost = np.inf
        best_path = np.inf
        start_time = time.time()

        for m in range(10):
            results = self.greedyLowerBound()
            path = results['soln']
            cost = results['cost']
            changed = True
            while changed:
                changed = False
                sizeNodes = len(self._scenario.getCities())
                for i in range(1, sizeNodes - 1):
                    for j in range(i + 1, sizeNodes):
                        new_path = self.twoOptSwap(path, i, j)
                        new_cost = new_path.costOfRoute()
                        if new_cost < cost:
                            cost = new_cost
                            changed = True
                            path = new_path
            if cost < best_cost:
                best_cost = cost
                best_path = path

        results['cost'] = best_cost
        results['time'] = time.time() - start_time
        results['count'] = 0
        results['soln'] = best_path
        return results

    def three_opt(self, start_time, time_allowance=60000.0):
        best_cost = np.inf
        best_path = np.inf
        start_time = time.time()

        for m in range(10):
            results = self.greedyLowerBound()
            path = results['soln']
            cost = results['cost']
            changed = True
            while changed:
                changed = False
                sizeNodes = len(self._scenario.getCities())
                for i in range(1, sizeNodes - 2):
                    for j in range(i + 1, sizeNodes - 1):
                        for k in range(j + 1, sizeNodes):
                            if (time.time() - start_time > time_allowance):
                                print("Best cost so far", cost)
                            new_path = self.threeOptSwap(path, i, j, k)
                            new_cost = new_path.costOfRoute()
                            if new_cost < cost:
                                cost = new_cost
                                changed = True
                                path = new_path
            if cost < best_cost:
                best_cost = cost
                best_path = path

        results['cost'] = best_cost
        results['time'] = time.time() - start_time
        final_time = time.time() - start_time
        print("TIME: ", final_time)
        results['count'] = 0
        results['soln'] = best_path
        return results

    def twoOptSwap(self, path, i, j):
        new_path = path.route[:i] + path.route[i:j][::-1] + path.route[j:]
        new_path_tsp = TSPSolution(new_path)
        return new_path_tsp

    def threeOptSwap(self, path, i, j, k):
        new_path = path.route[:i] + path.route[j:k] + path.route[i:j][::-1] + path.route[k:]
        new_path_tsp = TSPSolution(new_path)
        return new_path_tsp
