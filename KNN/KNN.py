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
        label.append(int(linesplit[-1]))

    return matrix, label

if __name__ == "__main__":
    path = 'KNN/dataset/datingTestSet2.txt'
    dataSet, label = txt2matrix(path)
    r = classify0([0, 1, 2], dataSet, label, 10)
    print(r)