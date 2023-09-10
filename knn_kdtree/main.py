import numpy as np
from KDtree import KNNpoints, insert
import dataSet
from collections import Counter
import time
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    Xpoints = dataSet.X_train_list
    Ypoints = dataSet.y_train
    XtestList = dataSet.X_test_list
    YtestList = dataSet.y_test

    for k_vecinos in range(1, 36):
        knn = KNNpoints(k_vecinos)
        knn.trainKNN(Ypoints)
        resultArr = []
        inicio_tiempo = time.time()
        tree_Predict = insert(root=None, points=Xpoints, depth=0, father=None)

        for x in range(len(XtestList)):
            tempTree = tree_Predict
            yPredict = knn.predictPoints(tempTree, XtestList[x])
            resultArr.append(yPredict)

        precision = accuracy_score(resultArr, YtestList)

        fin_tiempo = time.time()
        tiempo_procesamiento = (fin_tiempo - inicio_tiempo) * 1000
        print(f'Eficiencia para k={k_vecinos}: {precision* 100:.2f}% - Tiempo de procesamiento: {tiempo_procesamiento:.2f} ms')
