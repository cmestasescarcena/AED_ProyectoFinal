from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import dataSetApp
import KDtreeApp
from sklearn import preprocessing

app = Flask(__name__)

tree = KDtreeApp.kdTree(dataSetApp.X_train_list)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/data', methods = ["POST", "GET"])
def data():
    if request.method == "POST":
        age = request.form['age']
        restingBP = request.form['restingBP']
        cholesterol = request.form['cholesterol']
        maxHR = request.form['maxHR']
        oldPeak = request.form['oldPeak']
        chestPainType = request.form['flexRadioChestPain']
        sex = request.form['flexRadioSex']
        fastingBS = request.form['flexRadioFasting']
        restingECG = request.form['flexRadioResting']
        exerciseAngina = request.form['flexRadioExercise']
        st_slope = request.form['flexRadioSlope']

        if sex == 'M':
            sexM = True
        else:
            sexM = False

        if chestPainType == 'ATA':
            chestPainTypeATA = True
        else:
            chestPainTypeATA = False
        
        if chestPainType == 'NAP':
            chestPainTypeNAP = True
        else:
            chestPainTypeNAP = False
        
        if chestPainType == 'TA':
            chestPainTypeTA = True
        else:
            chestPainTypeTA = False
        
        if restingECG == 'Normal':
            restingECGNormal = True
        else:
            restingECGNormal = False
        
        if restingECG == 'ST':
            restingECGST = True
        else:
            restingECGST = False
        
        if exerciseAngina == 'Y':
            exerciseAnginaY = True
        else:
            exerciseAnginaY = False
        
        if st_slope == 'Flat':
            st_slopeFlat = True
        else:
            st_slopeFlat = False
        
        if st_slope == 'Up':
            st_slopeUp = True
        else:
            st_slopeUp = False

        columsdf = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']
        col_points = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']
        listInput = [[int(age), int(restingBP), int(cholesterol), int(fastingBS), int(maxHR), int(oldPeak), 0,  sexM, chestPainTypeATA, chestPainTypeNAP, chestPainTypeTA, restingECGNormal, restingECGST, exerciseAnginaY, st_slopeFlat, st_slopeUp]]
        
        dfInput = pd.DataFrame(listInput, columns=columsdf)
        
        dfInputSca = pd.DataFrame(dataSetApp.std_scaler.transform(dfInput), columns=columsdf)

        inputPoint = (dfInputSca.drop(['HeartDisease'], axis=1)).to_numpy().tolist()

        knn = KDtreeApp.KNNpoints(35, inputPoint[0], tree)

        df_temp = pd.DataFrame(knn.predict(),columns=col_points)
        neaPoint = [] #Array donde se almacenaran los resultados de los k puntos más cercanos
        for m in range(df_temp.__len__()):
            pointsDF = dataSetApp.total_df[
            (dataSetApp.total_df[col_points[0]] == df_temp[col_points[0]][m])&
            (dataSetApp.total_df[col_points[1]] == df_temp[col_points[1]][m])& 
            (dataSetApp.total_df[col_points[2]] == df_temp[col_points[2]][m])& 
            (dataSetApp.total_df[col_points[3]] == df_temp[col_points[3]][m])&
            (dataSetApp.total_df[col_points[4]] == df_temp[col_points[4]][m])&
            (dataSetApp.total_df[col_points[5]] == df_temp[col_points[5]][m])&
            (dataSetApp.total_df[col_points[6]] == df_temp[col_points[6]][m])&
            (dataSetApp.total_df[col_points[7]] == df_temp[col_points[7]][m])&
            (dataSetApp.total_df[col_points[8]] == df_temp[col_points[8]][m])&
            (dataSetApp.total_df[col_points[9]] == df_temp[col_points[9]][m])&
            (dataSetApp.total_df[col_points[10]] == df_temp[col_points[10]][m])&
            (dataSetApp.total_df[col_points[11]] == df_temp[col_points[11]][m])&
            (dataSetApp.total_df[col_points[12]] == df_temp[col_points[12]][m])&
            (dataSetApp.total_df[col_points[13]] == df_temp[col_points[13]][m])&
            (dataSetApp.total_df[col_points[14]] == df_temp[col_points[14]][m])
            ]['HeartDisease']
            neaPoint.append(pointsDF.to_list())

        ansPoint = max([[0], [1]], key=neaPoint.count)[0]

        if ansPoint == 1:
            resultData = "Possible Heart Disease"
        else:
            resultData = "No Possible Heart Disease"

        return redirect(url_for('result', res = resultData))
    else:
        return "bad request"

@app.route('/result/<string:res>')
def result(res):
    return render_template('home.html', prediction = res)

if __name__ == '__main__':
    app.run(debug=True)

