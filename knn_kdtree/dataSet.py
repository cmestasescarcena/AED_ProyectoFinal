import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

heart = pd.read_csv('AED_ProyectoFinal/knn_kdtree/heart.csv')

heart.describe() #Tipo de datos en el CSV, solo nos interesan los datos numericos
heart.isnull().sum() #Cantidad de datos nulos

RestingBP = heart[heart['RestingBP'] == 0 ] #Existe un dato incorrecto fila 449

heart = heart.drop(heart[(heart['RestingBP'] == 0)].index)

numerical = heart.select_dtypes(include=['int64', 'float64']).columns
categorical = heart.select_dtypes(include=['object']).columns

#Borrar valores atípicos de data frame
selected_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
df = heart[selected_columns]

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_df = df[((df>= lower_bound) & (df <= upper_bound)).all(axis=1)]

heart = pd.concat([heart, filtered_df])
heart.reset_index(drop=True, inplace=True)

#Preprosesing
Cholesterol = heart[heart['Cholesterol'] == 0]
heart.loc[heart['Cholesterol'] == 0, 'Cholesterol'] = np.nan
heart["Cholesterol"] = heart["Cholesterol"].fillna(heart["Cholesterol"].median())
Cholesterol1 = heart[heart['Cholesterol'] == 0]

heart = pd.get_dummies(heart,columns=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'],drop_first=True)
heart.fillna(0,inplace=True)

std_scaler = preprocessing.MinMaxScaler()

dataComp = heart
dataCompEsc = std_scaler.fit_transform(dataComp)
dataCompList = dataCompEsc.tolist()

total_df = pd.DataFrame(dataCompList, columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up'])

datos = total_df.drop(['HeartDisease'], axis=1)


X = datos
X_train,X_test = train_test_split(X,test_size=0.2,random_state=21)
print(X_test.info())