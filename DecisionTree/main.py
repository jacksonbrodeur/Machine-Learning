import pandas
import math
import queue

class Attribute:
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __str__(self):
        return "{}: {}".format(self.name, self.values)

const_attributes = [
    Attribute('size', ['Large', 'Medium', 'Small']),
    Attribute('occupied', ['High', 'Moderate', 'Low']),
    Attribute('price', ['Expensive', 'Normal', 'Cheap']),
    Attribute('music', ['Loud', 'Quiet']),
    Attribute('location', ['Talpiot', 'City-Center', 'Mahane-Yehuda', 'Ein-Karem', 'German-Colony']),
    Attribute('VIP', ['Yes', 'No']),
    Attribute('favorite beer', ['Yes', 'No']),
    Attribute('enjoy', ['Yes', 'No'])
]

SIZE = 0
OCCUPIED = 1
PRICE = 2
MUSIC = 3
LOCATION = 4
VIP = 5
FAVORITE_BEER = 6
ENJOY = 7

class Node:
    def __init__(self, attribute, children = {}):
        self.value = attribute
        self.children = children

def calculateEntropy(numPositive, numNegative):
    total = numPositive + numNegative
    try:
        pPositive = float(numPositive)/total
        pNegative = float(numNegative)/total
        return  pPositive * math.log2(1/pPositive) + pNegative * math.log2(1/pNegative)
    except ZeroDivisionError:
        return 0

def chooseBestAttribute(attributes, data):
    numPositive = len(data.loc[data['enjoy'] == 'Yes'].index)
    numNegative = len(data.loc[data['enjoy'] == 'No'].index)
    if(numPositive == 0 or numNegative == 0):
        return None

    entropy = calculateEntropy(numPositive, numNegative)
    maxInfoGain = 0
    bestAttr = None
    for attr in attributes:
        if attr.name == 'enjoy':
            continue
        entropyAfterSplit = 0
        for value in attr.values:
            valueData = data.loc[data[attr.name] == value]
            newNumPositive = len(valueData.loc[valueData['enjoy'] == 'Yes'].index)
            newNumNegative = len(valueData.loc[valueData['enjoy'] == 'No'].index)
            newTotal = newNumNegative + newNumPositive
            valueEntropy = calculateEntropy(newNumPositive, newNumNegative)
            entropyAfterSplit += valueEntropy * float(newTotal)/(numPositive + numNegative)

        if entropy - entropyAfterSplit > maxInfoGain:
            maxInfoGain = entropy - entropyAfterSplit
            bestAttr = attr

    return bestAttr

def buildDecisionTree(attributes, data):
    bestAttr = chooseBestAttribute(attributes, data)

    if bestAttr == None:
        numPositive = len(data.loc[data['enjoy'] == 'Yes'].index)
        numNegative = len(data.loc[data['enjoy'] == 'No'].index)
        decisionNode = Node(const_attributes[ENJOY])
        decisionNode.children = "Yes" if numPositive > numNegative else "No"
        return decisionNode
    
    splitDataSet = []
    for value in bestAttr.values:
        splitDataSet.append(data.loc[data[bestAttr.name] == value])

    newAttributes = [attr for attr in attributes if attr != bestAttr]
    children = []
    for ds in splitDataSet:
        children.append(buildDecisionTree(newAttributes, ds))

    return Node(bestAttr, {value: child for value, child in zip(bestAttr.values, children)})

def dtToString(node):
    myQueue = queue.Queue()
    myQueue.put(node)
    myQueue.put('\n')
    treeString = "Decision Tree: \n\t"
    lastWasNewline = False

    while not myQueue.empty():
        currentNode = myQueue.get()
        if currentNode == '\n':
            treeString += "\n\t"
            if not lastWasNewline:
                myQueue.put('\n')
            lastWasNewline = True

        else:
            lastWasNewline = False
            if currentNode.value.name == 'enjoy':
                treeString += currentNode.children + " "
            else:
                treeString += currentNode.value.name + " "
                for value in currentNode.value.values:
                    myQueue.put(currentNode.children[value])
                
    
    return treeString

def predict(input, dt):
    node = dt
    while not node.children in ["Yes", "No"]:
        node = node.children[input[node.value.name]]
    return node.children



headers = ['Num']
headers.extend([a.name for a in const_attributes])
table = pandas.read_table('./dt-data.txt', delimiter='[,\s]\s*', names=headers, skiprows=2, comment=';', engine='python')
dt = buildDecisionTree(const_attributes, table)
print(dtToString(dt))
input = {
    "size": "Large",
    "occupied": "Moderate",
    "price": "Cheap",
    "music": "Loud",
    "location": "City-Center",
    "VIP": "No",
    "favorite beer": "No"
}
print("Predicition for {} is {}".format(input, predict(input, dt)))

