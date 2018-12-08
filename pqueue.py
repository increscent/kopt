import random
import time

class PqueueArr:
    queue = []
    
    def __init__(self, size):
        self.queue = [None for _ in range (0, size)]
        self.size = size

    def insert(self, index, priority):
        self.queue[index] = priority

    def decreaseKey(self, index, priority):
        self.insert(index, priority)

    def deleteMin(self):
        minVal = 2**31
        minIndex = None
        for i in range(0, self.size):
            val = self.queue[i]
            if val != None and val < minVal:
               minVal = val
               minIndex = i

        if minIndex != None:
            self.queue[minIndex] = None

        return minIndex

class PqueueArr3:
    queue = []
    indeces = {}

    def __init__(self, size):
        self.queue = []
        self.indeces = {}

    # O(1)
    def insert(self, key, priority):
        self.queue.append(priority)
        self.indeces[key] = len(self.queue) - 1

    # O(1)
    def decreaseKey(self, key, priority):
        if not key in self.indeces:
            self.insert(key, priority)
        else:
            self.queue[self.indeces[key]] = priority

    # O(|V|)
    def deleteMin(self):
        if len(self.queue) == 0:
            return None

        # find minimum value in queue
        minVal = 2**31
        minIndex = None
        # runs |V| times
        for i in range(0, len(self.queue)):
            if self.queue[i] < minVal:
                minVal = self.queue[i]
                minIndex = i

        if minIndex == None:
            return None

        # find keys associated with minIndex and endIndex
        endIndex = len(self.queue) - 1
        result = None
        resultSwap = None
        # runs |V| times
        for key, value in self.indeces.items():
            if value == minIndex:
                result = key
            if value == endIndex:
                resultSwap = key

        # swap minIndex with endIndex (put minIndex value at the end)
        self.indeces[resultSwap] = self.indeces[result]
        self.queue[minIndex] = self.queue[endIndex]

        self.queue.pop()

        del self.indeces[result]

        return result

class PqueueArr2:
    queue = []
    indeces = {}
    fill = 0

    def priority(self, index):
        return self.queue[index] >> 32

    def key(self, index):
        return self.queue[index] & 0xFFFFFFFF

    def value(self, key, priority):
        priority = int(priority * 1024)

        return (priority << 32) | (key & 0xFFFFFFFF)

    def __init__(self, size):
        self.queue = [None for _ in range (0, size)]
        self.indeces = {}
        self.fill = 0

    def insert(self, key, priority):
        self.queue[self.fill] = self.value(key, priority)
        self.indeces[key] = self.fill

        i = self.fill
        while i > 0:
            if self.priority(i) > self.priority(i - 1):
                tmp = self.queue[i]
                self.queue[i] = self.queue[i - 1]
                self.queue[i - 1] = tmp

                self.indeces[self.key(i)] = i
                self.indeces[self.key(i - 1)] = i - 1

                i -= 1
            else:
                break

        self.fill += 1

    def decreaseKey(self, key, priority):
        if key not in self.indeces:
            self.insert(key, priority)
            return

        i = self.indeces[key]

        self.queue[i] = self.value(key, priority)

        while i < self.fill - 1:
            if self.priority(i) < self.priority(i + 1):
                tmp = self.queue[i]
                self.queue[i] = self.queue[i + 1]
                self.queue[i + 1] = tmp

                self.indeces[self.key(i)] = i
                self.indeces[self.key(i + 1)] = i

                i += 1
            else:
                break
                
    def deleteMin(self):
        if self.fill <= 0:
            return None

        result = self.key(self.fill - 1)

        self.fill -= 1

        return result

class PqueueHeap:
    heap = []
    fill = 0
    indeces = {}

    def priority(self, index):
        return self.heap[index][0]

    def value(self, index):
        return self.heap[index][1]

    def __init__(self):
        self.heap = []
        self.indeces = {}

    # O(log(|V|))
    def insert(self, value, priority):
        while len(self.heap) <= self.fill:
            self.heap.append(None)
        self.heap[self.fill] = (priority, value)
        self.indeces[value] = self.fill

        index = self.fill
        self.ascendHeap(index)
        
        self.fill += 1

    # O(log(|V|))
    def deleteMin(self):
        if self.fill == 0:
            return None

        self.fill -= 1

        result = self.heap[0]
        self.heap[0] = self.heap[self.fill]
        index = 0
        self.descendHeap(index)
        self.heap[self.fill] = None

        del self.indeces[result[1]]

        return result[1]

    # O(log(|V|))
    def decreaseKey(self, value, priority):
        if not value in self.indeces:
            self.insert(value, priority)
            return

        index = self.indeces[value]

        self.heap[index] = (priority, value)
        self.ascendHeap(index)


    # O(log(|V|))
    def ascendHeap(self, index):
        # move child up while smaller than its parent
        while index > 0:
            parentIndex = int((index - 1) / 2)
            if self.priority(parentIndex) > self.priority(index):
                tmp = self.heap[index]
                self.heap[index] = self.heap[parentIndex]
                self.heap[parentIndex] = tmp

                self.indeces[self.value(parentIndex)] = parentIndex
                self.indeces[self.value(index)] = index

                index = parentIndex
            else:
                break
        
    # O(log(|V|))
    def descendHeap(self, index):
        # move child down while larger than its parent
        while index < self.fill:
            childIndex1 = index * 2 + 1
            childIndex2 = index * 2 + 2
            if childIndex1 >= self.fill:
                break
            minChildIndex = childIndex1
            if childIndex2 < self.fill and self.priority(childIndex2) < self.priority(childIndex1):
                minChildIndex = childIndex2
            if self.priority(minChildIndex) < self.priority(index):
                tmp = self.heap[index]
                self.heap[index] = self.heap[minChildIndex]
                self.heap[minChildIndex] = tmp

                self.indeces[self.value(minChildIndex)] = minChildIndex
                self.indeces[self.value(index)] = index

                index = minChildIndex
            else:
                break

# testing
#size = 1000
#pqueue1 = PqueueArr(size)
#pqueue2 = PqueueHeap(size)
#values = []
#priorities = []
#
#for i in range(size):
#    priority = random.randint(1, 100000)
#    while priority in priorities:
#        priority = random.randint(1, 10000)
#    priorities.append(priority)
#    values.append((i, priority))
#    pqueue1.insert(i, priority)
#    pqueue2.insert(i, priority)
#
#
#sortedValues = sorted(values, key=lambda x: x[1])
#
#pqueue1Fail = 0
#pqueue2Fail = 0
#for i in range(size):
#    res1 = pqueue1.deleteMin()
#    res2 = pqueue2.deleteMin()
#    expected = sortedValues[i][0]
#
#    if res1 != expected:
#        pqueue1Fail += 1
#    if res2 != expected:
#        pqueue2Fail += 1
#
#print("PqueueArr fails: " + str(pqueue1Fail))
#print("PqueueHeap fails: " + str(pqueue2Fail))
