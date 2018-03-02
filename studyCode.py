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
