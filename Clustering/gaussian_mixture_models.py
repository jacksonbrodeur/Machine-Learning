import numpy as np
np.set_printoptions(threshold=np.inf)
from point import Point
import random
from scipy.stats import multivariate_normal

class Gaussian:

    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

    def __str__(self):
        return "mean: {}, covariance: {}".format(self.mean, self.covariance)

    def __repr__(self):
        return str(self)

def gaussian_mixture_models(x, y, k=3):
    dataMatrix = np.vstack((x,y))


    #initialization
    weights = np.random.rand(len(x), k)
    weights = weights / weights.sum(1)[:, np.newaxis]


    mean = None
    covariances = None

    for n in range(0, 20):

        #M-Step
        alpha = weights.sum(0) / dataMatrix.shape[1]
        mean = (np.matrix(dataMatrix) * np.matrix(weights))/(weights.sum(0))
        covariances = []
        for i in range(0, k):
            normalizedValues = np.matrix((dataMatrix - mean[:, i]))
            sigma = np.empty((2,2))
            for j in range(0, dataMatrix.shape[1]):
                sigma += weights[j, i] * (normalizedValues[:, j] * normalizedValues[:, j].transpose())
            covariances.append(sigma)

        #E-Step
        weights = np.empty((dataMatrix.shape[1], k))
        for i in range(0, k):
            weights[:, i] = multivariate_normal.pdf(dataMatrix.T, mean = mean[:, i].A1, cov = covariances[i]) * alpha[i]
        weights = weights / weights.sum(1)[:,np.newaxis]

    return mean, covariances
    



dataPoints = []
x = []
y = []

with open('clusters.txt', 'rb') as inputFile:
    for line in inputFile:
        line = line.decode("utf-8")
        line = line.strip()
        tempX, tempY = [float(val) for val in line.split(',')]
        x.append(tempX)
        y.append(tempY)

means = None
covariances = None
success = False

while not success:
    try:
        means, covariances = gaussian_mixture_models(x, y)
        success = True
    except:
        continue

for i in range(0, means.shape[1]):
    print("Gaussian {}".format(i))
    print("Mean: (x: {}, y: {})".format(means[:, i].A1[0], means[:, i].A1[1]))
    print("Covariance: {}".format(covariances[i]))