import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataPoints = np.empty([0,2])
labels = np.empty([0,1])

with open('linear-regression.txt', 'rb') as inputFile:
	for line in inputFile:
		line = line.decode('utf-8')
		line = line.strip()
		floatVals = [float(val) for val in line.split(',')]
		dataPoints = np.vstack([dataPoints, floatVals[0:2]])
		labels = np.vstack([labels, floatVals[2]])

dataPoints = np.append(np.ones([dataPoints.shape[0],1]), dataPoints, axis=1)

D = dataPoints.T

parameters = np.linalg.inv((D.dot(D.T))).dot(D).dot(labels)
print('Parameters:\n{}'.format(parameters))

xSurf, ySurf = [val / 100.0 for val in np.meshgrid(range(0, 100), range(0,100))]
zSurf = (parameters[1] * xSurf + parameters[2] * ySurf + parameters[0])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xSurf, ySurf, zSurf, color='r', alpha = 0.8)
ax.scatter(dataPoints[:, 1], dataPoints[:, 2], labels)
plt.show()