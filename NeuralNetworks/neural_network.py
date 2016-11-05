import numpy as np

np.set_printoptions(threshold=np.nan)

def read_pgm(file):
    assert file.readline().decode('utf-8') == 'P5\n'
    file.readline() #ignore
    (width, height) = [int(i) for i in file.readline().decode('utf-8').split()]
    depth = int(file.readline().decode('utf-8'))
    assert depth <= 255

    #raster = np.empty([0, 0])
    pixelValues = []
    # print('{}, {}'.format(width, height))
    for i in range(height):
        row = []
        for j in range(width):
            #row.append(ord(file.read(1)))
            pixelValues.append(ord(file.read(1)))
        #raster = np.vstack([raster, row])
    return pixelValues

def sigmoid(s):
    return 1/(1 + np.exp(-s))

trainingFiles = []
with open('downgesture_train.list', 'rb') as trainingList:
    for line in trainingList:
        line = line.decode('utf-8')
        line = line.strip()
        trainingFiles.append(line)
testFiles = []
with open('downgesture_test.list', 'rb') as testList:
    for line in testList:
        line = line.decode('utf-8')
        line = line.strip()
        testFiles.append(line)
trainingImages = [read_pgm(open(file, 'rb')) for file in trainingFiles]
testingImages = [read_pgm(open(file, 'rb')) for file in testFiles]

d0 = 961
d1 = 101
d2 = 1
w1 = (np.random.rand(d0, d1) - 0.5)
w2 = (np.random.rand(d1, d2) - 0.5)
learningRate = 0.1

# Train the Neural Network
# for n in range(len(trainingFiles)):
for n in range(0, 2000):
    k = n % len(trainingFiles)
    print(n)
    image = [1] + trainingImages[k]
    label = 1 if 'down' in trainingFiles[k] else 0

    # Feed forward to calculate x values in nodes of NN
    x1 = np.empty(d1)
    x1[0] = 1
    for j in range(1, d1):
        sum = 0
        for i in range(d0):
            sum += w1[i, j] * image[i]
        x1[j] = sigmoid(sum)

    x2 = np.empty(d2)
    for j in range(d2):
        sum = 0
        for i in range(d1):
            sum += w2[i, j] * x1[i]
        x2[j] = sigmoid(sum)

    # Backwards to calculate delta values
    deltaL = [2 * (x2[0] - label) * (1 - (x2[0] ** 2))]
    delta1 = np.empty(d1)
    for i in range(d1):
        sum = 0
        for j in range(1, d2):
            sum += w2[i, j] * deltaL[j-1]
        delta1[i] = (1 - x1[i] ** 2) * sum

    delta0 = np.empty(d0)
    for i in range(d0):
        sum = 0
        for j in range(1, d1):
            sum += w1[i,j] * delta1[j]
        delta0[i] = (1 - image[i] ** 2) * sum

    # update weights
    for i in range(d0):
        for j in range(d1):
            w1[i,j] = w1[i,j] - learningRate * image[i] * delta1[j]
    for i in range(d1):
        for j in range(d2):
            w2[i, j] = w2[i,j] - learningRate * x1[i] * deltaL[j]

# Test the NN with the test data
numCorrect = 0
numIncorrect = 0
error = 0
for n in range(len(testFiles)):
    image = [1] + testingImages[n]
    fileName = testFiles[n]
    label = 1 if 'down' in fileName else 0

    x1 = np.empty(d1)
    x1[0] = 1
    for j in range(1, d1):
        sum = 0
        for i in range(d0):
            sum += w1[i,j] * image[i]
        x1[j] = sigmoid(sum)

    x2 = np.empty(d2)
    for j in range(d2):
        sum = 0
        for i in range(d1):
            sum += w2[i,j] * x1[i]
        x2[j] = sigmoid(sum)
    prediction = x2[0]
    threshold = 0.75
    outcome = 'down' if prediction >= threshold else 'other'
    if label == 1 and prediction >= threshold:
        numCorrect += 1
    elif label == 0 and prediction < threshold:
        numCorrect += 1
    else :
        numIncorrect += 1
    error += (label - prediction) ** 2

    print('{}: {}, {}'.format(fileName, outcome, prediction))

print('Correct predictions: {}'.format(numCorrect))
print('Incorrect predictions: {}'.format(numIncorrect))
print('Cumulative error: {}'.format(error))