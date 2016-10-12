import math
canPlot = True
try:
    import matplotlib.pyplot as plt
except:
    print("Install matplot lib to see clusters plotted\n")
    canPlot = False

def getFurthestPoints(points):
    maxDistance = 0
    maxPoint = None
    for point, distance in points.items():
        if distance > maxDistance:
           maxDistance = distance
           maxPoint = point
        elif distance == maxDistance and (point[0] < maxPoint[0] and point[0] < maxPoint[1]):
            maxDistance = distance
            maxPoint = point

    return maxPoint


distances = {}
numObjects = 0
with open('fastmap-data.txt', 'rb') as inputFile:
    for line in inputFile:
        line = line.decode("utf-8")
        line = line.strip()
        id1, id2, distance = [int(val) for val in line.split('\t')]
        numObjects = max(numObjects, id1, id2)
        distances[(id1, id2)] = distance
        distances[(id2, id1)] = distance

words = []
with open('fastmap-wordlist.txt', 'rb') as inputFile:
    for line in inputFile:
        line = line.decode("utf-8")
        line = line.strip()
        words.append(line)

oa, ob = getFurthestPoints(distances)

dataPoints = {}
dataPoints[oa] = 0
dataPoints[ob] = distances[(oa, ob)]

for i in range(1, numObjects + 1):
    if i == oa or i == ob:
        continue
    else:
        dataPoints[i] = (distances[(i,oa)]**2 + distances[(oa, ob)]**2 - distances[(i, ob)] ** 2) / (2 * distances[(oa, ob)])

newDistances = {}
for i in range(1, numObjects + 1):
    for j in range(1, numObjects + 1):
        if i == j:
            continue
        else:
            newDist = math.sqrt(distances[(i, j)] ** 2 - (dataPoints[i] - dataPoints[j]) ** 2)
            newDistances[(i,j)] = newDist
            newDistances[(j,i)] = newDist

oa, ob = getFurthestPoints(newDistances)

dataPointsY = {}
dataPointsY[oa] = 0
dataPointsY[ob] = distances[(oa, ob)]

for i in range(1, numObjects + 1):
    if i == oa or i == ob:
        continue
    else:
        dataPointsY[i] = (newDistances[(i, oa)]**2 + newDistances[(oa, ob)]**2 - newDistances[(i, ob)]**2) / (2 * newDistances[(oa, ob)])

x = []
y = []
for i in range(1, numObjects + 1):
    x.append(dataPoints[i])
    y.append(dataPointsY[i])


plt.scatter(x, y)
for label, px, py in zip(words, x, y):
    plt.annotate(label, xy=(px, py))
plt.show()