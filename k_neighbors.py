# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:09:30 2015

@author: Haley
"""
# YOU ARE GIVEN THIS CLASS
class Example(object):
    def __init__(self, name, featureVec, label = None):
        self.name = str(name)
        self.featureVec = featureVec
        self.label = label
    def distance(self, other, pwr = 2):
        dist = 0.0
        for i in range(len(self.featureVec)):
            dist += abs(self.featureVec[i] - other.featureVec[i])**pwr
        return dist**(1.0/pwr)
    def getFeatures(self):
        return self.featureVec[:]
    def getLabel(self):
        return self.label
    def __str__(self):
        return self.name + ',' + str(self.featureVec)\
               + ', ' + str(self.label)

# YOU ARE GIVEN THIS FUNCTION
def findKNearest(example, exampleSet, k):
    kNearest, distances = [], []
    #Build lists containining first k examples, and their distances
    for i in range(k):
        kNearest.append(exampleSet[i])
        distances.append(example.distance(exampleSet[i]))
    maxDist = max(distances) #Get maximum distance
    #Look at examples not yet considered
    for e in exampleSet[k:]:
        dist = e.distance(example)
        if dist < maxDist:
            maxIndex = distances.index(maxDist)
            kNearest[maxIndex] = e
            distances[maxIndex] = dist
            maxDist = max(distances)      
    return kNearest, distances
    
# IMPLEMENT THIS FUNCTION
def KNearestClassify(training, example, k, weighted = False):
    """
    training: a list of elements of type Example
    example: a value of type Example
    k: an int > 0
    weighted: a Boolean

    Uses k-nearest neighbors (with Euclidean distance) to find 
    the label for example, from possible labels of True or False.
    The neighbors are chosen from training.
    If weighted == False, each of the k neighbors gets a vote of 1.
    If weighted == True, each of the k neighbors gets a vote equal
    to the inverse of its distance from example. Assume distances 
    between all pairs of examples are non-zero.

    Returns a Boolean representing the label of example 
    (True or False) or raises a ValueError if there is a tie.
    """
    
    ex, distances = findKNearest(example, training, k)
    truenum = 0
    falsenum = 0
    
    if weighted == False:
        for i in xrange(len(distances)):
            if ex[i].getLabel() == True:
                truenum += 1
            else:
                falsenum += 1
    else:
        for i in xrange(len(distances)):
            if ex[i].getLabel() == True:
                truenum += 1.0 / distances[i]
            else:
                falsenum += 1.0 / distances[i]
    
    if truenum > falsenum:
        return True
    else:
        return False
        
    
    
ex1 = Example('ex1', [1], True)
ex2 = Example('ex2', [1], True)
ex3 = Example('ex3', [1], True)
ex4 = Example('ex4', [0], False)
examples = [ex1, ex2, ex3, ex4]


ex0 = Example('test', [0.01])
print KNearestClassify(examples, ex0, 3, False)