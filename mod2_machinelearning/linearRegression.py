#   Modelo de regresión lineal entre el rating de un anime y la visibilidad que tiene entre una cantidad de usuarios
#   Dataset de: https://www.kaggle.com/datasets/alancmathew/anime-dataset
#   @author: Alejandro López Hernández A01733984
#   Comportamiento esperado: A mayor rating o calificación del anime, mayor será la cantidad de usuarios que lo habran visto
#   Animes (TV, Películas y OVAs) empleados: 11985

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
X = np.array(df['rating'])

#determinación de la variable Y como la cantidad de usuarios que han dicho haber visto el anime
Y = np.array(df['watched'])



#función del gradiente descendiente para la regresión lineal
def GradientDesc(m,c,alpha,epochs,X,Y):
	epoch = 0
	n = float(len(X))

	while epoch < epochs: 
		Y_pred = m * X + c # we calculate the hyp
		D_m = (-2/n) * sum(X * ( Y - Y_pred )) # we calculate derivate of D_m
		D_c = (-2/n) * sum(Y - Y_pred) # we calculate derivate of D_c
		m = m -alpha*D_m # we update the value of m and c
		c = c -alpha*D_c
		epoch += 1
		if( epoch == epochs ):
			print("pendiente: ", m, " intercepto: ", c)
	#se retorna la pendiente y el intercepto
	return(m,c)

#funcion de predicción dada una pendiente, un intercepto y un rating de anime
def predict(m,c,anime_input):
	y = m*anime_input+c
	print(f"Rating del anime: {anime_input} Usuarios que lo han visto: {int(y)}")

	return
# Variables a emplear en el modelo (m -> pendiente, c -> el intercepto, l -> el learning rate, epochs -> epocas del modelo)
m = 0
c = 0 
L = .0001
epochs = 1000

#obtencion de la pendiente y el intercepto
m, c = GradientDesc(m,c,L,epochs,X,Y)

#la prediccion de y para la linea de regresion de la grafica
y_pred = m*X + c

#predicciones, con el rating del anime como input
predict(m,c,3.7)
predict(m,c,4.5)
predict(m,c,2.9)
predict(m,c,1.3)
predict(m,c,0.5)


#graficacion de los datos 
plt.scatter(X,Y)
plt.plot([min(X),max(X)],[min(y_pred),max(y_pred)], color='red')
plt.xlabel("Anime rating")
plt.ylabel("People watched")
plt.title("Anime Rating vs Watched")
plt.show()