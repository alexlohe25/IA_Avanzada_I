#   Modelo de regresión lineal entre los votos recibidos de un anime y su rating o calificacion
#   Dataset de: https://www.kaggle.com/datasets/alancmathew/anime-dataset
#   @author: Alejandro López Hernández A01733984
#   Comportamiento esperado: A mayor cantidad de votos, mayor será la calificación
#   Animes (TV, Películas y OVAs) empleados: 11985

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso


#Obtención de los datos a partir de un csv con pandas
columns = ["title","mediaType","eps","duration","ongoing","startYr","finishYr","sznOfRelease","description","studios","tags","contentWarn","watched","watching","wantWatch","dropped","rating","votes"]
df = pd.read_csv('mod2_machinelearning/anime.csv', names = columns)

#Limpieza de datos, considerando que no puede haber datos vacíos en dichas columnas
df = df.dropna(subset=['rating', 'votes'])

#conversion de columnas principales a tipo float
df['votes']= df['votes'][1:].astype(int)
df['rating'] = df['rating'][1:].astype(float)

#Los animes deben tener al menos 1 voto
df = df[df['votes'] > 0]

print(df)

#determinación de la variable correspondiente al eje X como los votos del anime
X = np.array(np.log(df['votes'])).reshape(-1,1)
#determinación de la variable Y como la calificación del anime
Y = np.array(df['rating']).reshape(-1,1)

#division del data set en train y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

#llamada a la regresión lineal de scikit learn.
regr = LinearRegression()
regr.fit(X_train, Y_train)
#coeficiente e intercepto
print("Coefficient: ", regr.coef_)
print("Intercept: ", regr.intercept_)
#predicciones
print("Predicitons:")
custom_pred = [[1234], [2345], [6349], [42368], [923], [14]]

for i in custom_pred:
    print(f"Votos: {i} Rating aproximado: {regr.predict(np.log([i]))}")

#obtencion del error de prediccion en test y train
Y_pred = regr.predict(X_test)
Y_pred_train = regr.predict(X_train)
cross_val = abs(cross_val_score(regr, X_train, Y_train, cv=10, scoring = "r2").mean())

print("MSE test: ",mean_squared_error(Y_test, Y_pred))
print("MSE train: ",mean_squared_error(Y_train, Y_pred_train))

print("Model score test: ", r2_score(Y_test, Y_pred))
print("Model score train: ", r2_score(Y_train, Y_pred_train	))

print("Cross validation: ", cross_val)

model_lasso = Lasso(alpha = 0.01)
model_lasso.fit(X_train, Y_train)
pred_train_lasso = model_lasso.predict(X_train)
pred_test_lasso = model_lasso.predict(X_test)

print("MSE in Lasso train: ", mean_squared_error(Y_train, pred_train_lasso))
print("MSE in Lasso test: ", mean_squared_error(Y_test, pred_test_lasso))

print("Lasso score train: ", r2_score(Y_train, pred_train_lasso))
print("Lasso score test: ", r2_score(Y_test, pred_test_lasso))



#plot
figure, axis = plt.subplots(2,2)
#TEST
#regresion
axis[0,0].scatter(X_test, Y_test, alpha = 0.5)
axis[0,0].plot(X_test, Y_pred, color='red', label = "MSE: " + str(mean_squared_error(Y_test, Y_pred)))
axis[0,0].set_title("Anime Votes vs Rating (test data)")
axis[0,0].set(xlabel = 'Votes in ln(x)', ylabel = 'Anime rating')
axis[0,0].legend()

#varianza
axis[0,1].scatter(range(len(X_test)), Y_test, alpha = 0.3, label = 'Real data')
axis[0,1].scatter(range(len(X_test)), Y_pred, color='orange',alpha = 0.1, label = 'Predicted data')
axis[0,1].set_title("Real test data vs Predicted test data")
axis[0,1].set(xlabel = 'Votes in ln(x)', ylabel = 'Anime rating')
axis[0,1].legend()

#TRAIN
#regresion
axis[1,0].scatter(X_train, Y_train, alpha = 0.3)
axis[1,0].plot(X_train, Y_pred_train, color='red',label = "MSE: " + str(mean_squared_error(Y_train, Y_pred_train)))
axis[1,0].set_title("Anime Votes vs Rating (train data)")
axis[1,0].set(xlabel = 'Votes in ln(x)', ylabel = 'Anime rating')
axis[1,0].legend()

#varianza
axis[1,1].scatter(range(len(X_train)), Y_train, alpha = 0.3, label = 'Real data')
axis[1,1].scatter(range(len(X_train)), Y_pred_train, color='orange',alpha = 0.3, label = 'Predicted data')
axis[1,1].set_title("Real train data vs Predicted train data")
axis[1,1].set(xlabel = 'Votes in ln(x)', ylabel = 'Anime rating')
axis[1,1].legend()
plt.show()

