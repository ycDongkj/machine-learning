import numpy as np
import operator

def classify0(inx, dataSet, label, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inx, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sumMat = sqDiffMat.sum(axis=1)
    sortedDistancePos = sumMat.argsort()
    classCount = {}

    for i in range(k):
        curlabel = label[sortedDistancePos[i]]
        classCount[curlabel] = classCount.get(curlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key = lambda item:item[1],reverse=True)
    return sortedClassCount[0][0]


def txt2matrix(txtfile):
    with open(txtfile, 'r') as f:
        content = f.readlines()
    lenOfLines = len(content)
    matrix = np.zeros((lenOfLines, 3))
    label = []

    for i, line in enumerate(content):
        linesplit = line.split('\t')
        matrix[i] = linesplit[0:-1]
        label.append(linesplit[-1].strip())

    return matrix, label


def autoNorm(dataset):
    mn = dataset.min(0)
    mx = dataset.max(0)
    ranges = mx - mn
    datasetNum = dataset.shape[0]

    dataset = (dataset -np.tile(mn, (datasetNum, 1)))/np.tile(ranges, (datasetNum,1))

    return dataset, mn, ranges

def datingClassTest():
    path = './KNN/dataset/datingTestSet.txt'
    testRotio = 0.1

    dataSet,label = txt2matrix(path)
    dataNum = dataSet.shape[0]
    randPerm = list(np.random.permutation(dataNum))
    dataSet = dataSet[randPerm]
    label = [label[i] for i in randPerm]

    dataSet, mn, ranges = autoNorm(dataSet)

    trainPos = (int)(dataNum*(1-testRotio))
    trainDataset = dataSet[0:trainPos]
    trainLabel = label[0:trainPos]
    testDataset = dataSet[trainPos:]
    testLabel = label[trainPos:]

    trueNum = 0
    for i,sample in enumerate(testDataset):
        result = classify0(sample, trainDataset, trainLabel, 10)
        if result == testLabel[i]:
            trueNum = trueNum + 1
    print('accuracy:'+str(trueNum/(dataNum-trainPos)))


if __name__ == "__main__":
    datingClassTest()