import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def gradient(w, dataPoints, labels):
	grad = 0
	numPoints = dataPoints.shape[0]
	for i in range(numPoints):
		x = dataPoints[i]
		y = labels[i]
		grad += (1 / math.exp((y * w.T).dot(dataPoints[i]))) * y * x
	return grad / numPoints

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

iterations = range(0, 7000)
eta = 0.05

for i in iterations:
	grad = gradient(parameters, dataPoints, labels)
	grad = np.reshape(grad, (grad.shape[0], 1))
	parameters -= eta * (grad / np.linalg.norm(grad))

print("Parameters:\n{}".format(parameters))
