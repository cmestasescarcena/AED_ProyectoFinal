import pandas as pd
import KDtree
import dataSet

if __name__ == '__main__':
    #Creacion de KD tree
    kd_Tree = KDtree.kdTree(dataSet.X_train_list)

    #K-vecinos
    kVecinos = [1, 2]

    #Columnas de data frame 
    col_df = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']
    col_points = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']
    
    arrSol = []
    arrSol1 = []

    for i in range(len(kVecinos)):
        for j in range(len(dataSet.X_test_list)):
            knn = KDtree.KNNpoints(kVecinos[i], dataSet.X_test_list[j], kd_Tree)
            arrSol1.append(knn.predict())
        arrSol.append(arrSol)
    
    print(len(arrSol))


"""
    resultClass = []

    for i in range(len(knn.predict())):
        resultClass1 = dataSet.total_df[dataSet.total_df[col_df[0]] == knn.predict()[i][0]]
        resultClass2 = resultClass1[resultClass1[col_df[1]] == knn.predict()[i][1]]['HeartDisease']
        resultClass.append(resultClass2.to_list())
    
    #print("Clase de consulta = ", max([0, 1], key=resultClass.count))
    #print(resultClass)

    """