#   Modelo de regresión lineal entre el rating de un anime y la visibilidad que tiene entre una cantidad de usuarios
#   Dataset de: https://www.kaggle.com/datasets/alancmathew/anime-dataset
#   @author: Alejandro López Hernández A01733984
#   Comportamiento esperado: A mayor rating o calificación del anime, mayor será la cantidad de usuarios que lo habran visto
#   Animes (TV, Películas y OVAs) empleados: 11985

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#Obtención de los datos a partir de un csv con pandas
columns = ["title","mediaType","eps","duration","ongoing","startYr","finishYr","sznOfRelease","description","studios","tags","contentWarn","watched","watching","wantWatch","dropped","rating","votes"]
df = pd.read_csv('mod2_machinelearning/anime.csv', names = columns)

#conversion de columnas principales a tipo float
df['watched']= df['watched'][1:].astype(float)
df['rating'] = df['rating'][1:].astype(float)

#Limpieza de datos, considerando que no puede haber datos vacíos en dichas columnas
df = df.drop(df[df.rating.isnull()].index)
df = df.drop(df[df.watched.isnull()].index)


#Los usuarios deben haber visto el anime por lo menos una vez
df = df[df['watched'] > 0]

print(df)

#determinación de la variable correspondiente al eje X como el rating del anime
X = np.array(df['rating']).reshape(-1,1)

#determinación de la variable Y como la cantidad de usuarios que han dicho haber visto el anime
Y = np.array(df['watched']).reshape(-1,1)

#division del data set en train y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)

#llamada a la regresión lineal de scikit learn, con la opción de fit_intercept en falso. Esto con tal de que el intercepto arranque
#en Y = 0, y así la regresión pueda ser más acertada respecto a mis datos; puesto que sólo se consideran datos positivos.
regr = LinearRegression(fit_intercept = False).fit(X_train,Y_train)
#coeficiente e intercepto
print("Coefficient: ",regr.coef_)
print("Intercept: ",regr.intercept_)

#predicciones
print("Predicitons:")
custom_pred = [[3.7], [4.5], [2.9], [1.3], [0.5]]

for i in custom_pred:
    print(f"Rating del anime: {i} Usuarios que lo han visto: {regr.predict([i])}")

#obtencion del error de prediccion en test y train
Y_pred = regr.predict(X_test)
Pred_error_test = Y_pred - Y_test
Y_pred_train = regr.predict(X_train)
Pred_error_train = Y_pred_train - Y_train

#plot de la regresión
figure, axis = plt.subplots(2,2)

axis[0,0].scatter(X_test, Y_test)
axis[0,0].plot(X_test, Y_pred, color='red')
axis[0,0].set_title("Anime Rating(x) vs Watched(y) (test data)")

axis[0,1].hist(Pred_error_test)
axis[0,1].set_title('Histogram of test prediction error')
axis[0,1].set_xlim(-60000, 60000)

axis[1,0].scatter(X_train, Y_train)
axis[1,0].plot(X_train, Y_pred_train, color ='red')
axis[1,0].set_title("Anime Rating(x) vs Watched(y) (train data)")

axis[1,1].hist(Pred_error_train)
axis[1,1].set_title('Histogram of train prediction error')
axis[1,1].set_xlim(-60000, 60000)

plt.show()