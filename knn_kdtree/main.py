import pandas as pd
from statistics import mode
import KDtree
import dataSet
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    #Creacion de KD tree
    kd_Tree = KDtree.kdTree(dataSet.X_train_list)

    #K-vecinos
    kVecinos = [35]

    #Columnas de data frame 
    col_df = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']
    col_points = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']


    knnArr = [] #Array k puntos
   
    for i in range(len(kVecinos)):
        knnArr_point = []
        for j in range(len(dataSet.X_test_list)):
         
            knn = KDtree.KNNpoints(kVecinos[i], dataSet.X_test_list[j], kd_Tree)
            knnArr_point.append(knn.predict())

        knnArr.append(knnArr_point)
    

    heartDisease = []

    for k in range(len(kVecinos)):
        ansPoint = []
        for n in range(len(dataSet.X_test_list)):
            df_temp = pd.DataFrame(knnArr[k][n],columns=col_points)
            neaPoint = [] #Array donde se almacenaran los resultados de los k puntos más cercanos

            for m in range(df_temp.__len__()):
                pointsDF = dataSet.total_df[
                (dataSet.total_df[col_points[0]] == df_temp[col_points[0]][m])&
                (dataSet.total_df[col_points[1]] == df_temp[col_points[1]][m])& 
                (dataSet.total_df[col_points[2]] == df_temp[col_points[2]][m])& 
                (dataSet.total_df[col_points[3]] == df_temp[col_points[3]][m])&
                (dataSet.total_df[col_points[4]] == df_temp[col_points[4]][m])&
                (dataSet.total_df[col_points[5]] == df_temp[col_points[5]][m])&
                (dataSet.total_df[col_points[6]] == df_temp[col_points[6]][m])&
                (dataSet.total_df[col_points[7]] == df_temp[col_points[7]][m])&
                (dataSet.total_df[col_points[8]] == df_temp[col_points[8]][m])&
                (dataSet.total_df[col_points[9]] == df_temp[col_points[9]][m])&
                (dataSet.total_df[col_points[10]] == df_temp[col_points[10]][m])&
                (dataSet.total_df[col_points[11]] == df_temp[col_points[11]][m])&
                (dataSet.total_df[col_points[12]] == df_temp[col_points[12]][m])&
                (dataSet.total_df[col_points[13]] == df_temp[col_points[13]][m])&
                (dataSet.total_df[col_points[14]] == df_temp[col_points[14]][m])
                ]['HeartDisease']
                neaPoint.append(pointsDF.to_list())
            
            ansPoint.append(max([[0], [1]], key=neaPoint.count)[0])

        heartDisease.append(ansPoint)


    y_respt = None
    for l in range(len(kVecinos)):
        if y_respt is None:
            y_respt = pd.DataFrame(heartDisease[l])
        else:
            y_respt[l] = heartDisease[l]
    
    for a in range(len(kVecinos)):
        print('Eficiencia para k =',kVecinos[a],' ', accuracy_score(y_respt[a], dataSet.y_test))
