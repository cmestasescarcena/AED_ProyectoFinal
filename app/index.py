from flask import Flask, render_template, request, url_for
import pandas as pd
from knn_kdtree import dataSet
from sklearn import preprocessing

app = Flask(__name__)

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

        listInput = [[int(age), int(restingBP), int(cholesterol), int(fastingBS), int(maxHR), int(oldPeak), sexM, chestPainTypeATA, chestPainTypeNAP, chestPainTypeTA, restingECGNormal, restingECGST, exerciseAnginaY, st_slopeFlat, st_slopeUp]]
        
        dfInput = pd.DataFrame(listInput, columns=['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up'])
        
        dfInputSca = dataSet.std_scaler.transform(dfInput)

        return f"{dfInputSca}"
    else:
        return "bad request"

if __name__ == '__main__':
    app.run(debug=True)

