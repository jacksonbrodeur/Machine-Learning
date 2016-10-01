import random as rand
from functools import reduce
import math

canPlot = True
try:
    import matplotlib.pyplot as plt
except:
    print("Install matplot lib to see clusters plotted\n")
    canPlot = False

class Point:
    """ Point class represents and manipulates x,y coords. """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "x: {}, y: {}".format(self.x, self.y)

    def __repr__(self):
        return str(self)

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

def calculateMaxMin(dataPoints):
    maxX = reduce(lambda a,b: a if a.x > b.x else b, dataPoints, dataPoints[0])
    maxY = reduce(lambda a,b: a if a.y > b.y else b, dataPoints, dataPoints[0])
    maxValue = max(maxX.x, maxY.y)
    minX = reduce(lambda a,b: a if a.x < b.x else b, dataPoints, dataPoints[0])
    minY = reduce(lambda a,b: a if a.y < b.y else b, dataPoints, dataPoints[0])
    minValue = min(minX.x, minY.y)
    return (maxValue, minValue)


def nearestPoint(point, otherPoints):
    minDistance = point.distance(otherPoints[0])
    minIndex = 0
    for index, otherPoint in enumerate(otherPoints):
        distance = point.distance(otherPoint)
        if distance < minDistance:
            minDistance = distance
            minIndex = index

    return minIndex

def recalculateCentroids(clusters):
    centroids = []
    for cluster in clusters:
        centroidX = sum([p.x for p in cluster]) / len(cluster)
        centroidY = sum([p.y for p in cluster]) / len(cluster)
        centroids.append(Point(centroidX, centroidY))
    return centroids



def k_means(dataPoints, k = 3):
    max, min = calculateMaxMin(dataPoints)
    centroids = []
    for i in range(0, k):
       centroids.append(Point(rand.uniform(min, max), rand.uniform(min, max)))
    clusters = []
    
    for i in range(0, 20):
        clusters = [[] for c in centroids]
        for point in dataPoints:
            index = nearestPoint(point, centroids)
            clusters[index].append(point)

        centroids = recalculateCentroids(clusters)

    return (centroids, clusters)
    


dataPoints = []

with open('clusters.txt', 'rb') as inputFile:
    for line in inputFile:
        line = line.decode("utf-8")
        line = line.strip()
        x, y = [float(val) for val in line.split(',')]
        dataPoints.append(Point(x,y))

centroids, clusters = k_means(dataPoints)

for i in range(0, len(centroids)):
    print("Centroid: {}, Points: {}\n\n\n\n\n".format(centroids[i], clusters[i]))

if canPlot:
    for cluster in clusters:
        plt.plot([p.x for p in cluster], [p.y for p in cluster], '.')
    for centroid in centroids:
        plt.plot(centroid.x, centroid.y, 'k+')
    plt.show()
