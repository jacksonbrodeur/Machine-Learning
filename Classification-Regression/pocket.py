import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getViolatedConstraint(x, y, w):
	constraints = x.dot(w)
	violatedIndex = None
	violatedLabel = None
	count = 0
	for index, (constraint, label) in enumerate(zip(constraints, y)):
		if (label == 1 and constraint < 0) or (label == -1 and constraint >= 0):
			violatedIndex = index
			violatedLabel = label
			count += 1
	if(not violatedLabel):
		return None
	return (violatedIndex, violatedLabel, count)

dataPoints = np.empty([0,3])
labels = np.empty([0,1])

with open('classification.txt', 'rb') as inputFile:
	for line in inputFile:
		line = line.decode('utf-8')
		line = line.strip()
		floatVals = [float(val) for val in line.split(',')]
		dataPoints = np.vstack([dataPoints, floatVals[0:3]])
		labels = np.vstack([labels, floatVals[4]])

dataPoints = np.append(np.ones([dataPoints.shape[0],1]), dataPoints, axis=1)
parameters = np.random.rand(dataPoints.shape[1], 1) - 0.5


violated = getViolatedConstraint(dataPoints, labels, parameters)
iterations = range(0, 7000)
bestParams = []
leastViolatedConstraints = np.inf
violatedConstraints = []
alpha = 0.01
for val in iterations:
	print(val)
	index, y, numViolated = violated
	print(numViolated)
	violatedConstraints.append(numViolated)
	point = dataPoints[index, ]
	point = np.reshape(point, (point.shape[0], 1))
	# alpha = 0.0001 * numViolated
	parameters = parameters + alpha * y * point
	if(numViolated < leastViolatedConstraints):
		bestParams = parameters
		leastViolatedConstraints = numViolated
	violated = getViolatedConstraint(dataPoints, labels, parameters)

print(bestParams)
print(leastViolatedConstraints)

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
xSurf, ySurf = [val / 100.0 for val in np.meshgrid(range(-20, 100), range(-20,100))]
zSurf = (bestParams[1] * xSurf + bestParams[2] * ySurf + bestParams[0]) / (-1 * bestParams[3])
colors = ['r' if label < 0 else 'b' for label in labels]
ax.plot_surface(xSurf, ySurf, zSurf, color='w', alpha = 0.8)
ax.scatter(dataPoints[:, 1], dataPoints[:, 2], dataPoints[:, 3], c = colors)

plt.figure(2)
plt.step(iterations, violatedConstraints)
plt.show()

