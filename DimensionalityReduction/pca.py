import numpy as np
np.set_printoptions(threshold=np.nan)

dataPoints = np.loadtxt('pca-data.txt')
dim = dataPoints.shape[1]

mean = np.mean(dataPoints, 0)
normalizedData = dataPoints - mean
# print(type(normalizedData))
# print(normalizedData)

sigma = np.zeros((3,3))

for i in range(0, dim):
	sigma += np.outer(normalizedData[i, :], normalizedData[i,:].T)

eigenvalues, eigenvectors = np.linalg.eig(sigma)
# sort eigenvalues in decreasing order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

uTr = eigenvectors[:, 0: dim - 1]

for i in range(0, dim - 1):
	print(uTr[:, i])