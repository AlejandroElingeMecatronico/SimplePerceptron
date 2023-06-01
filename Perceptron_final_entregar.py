#!/usr/bin/env python
# coding: utf-8

"""Autor: Hector Alejandro Pereyra Vargas
Matricula: EIM22
Materia: Inteligencia Artificial
Profesor: Ingeniero Tulio Gensana.
Consigna: Realización de un Perceptron Simple para prediccion de especies de Flor IRIS."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Cargamos el dataset y observamos los 10 primeros valores
df = pd.read_csv(r'C:\Users\alepe\OneDrive\Escritorio\CSV\Iris.csv')
print(df.head(10))

#Informacion del del dataset: 
print(df.info())

#Estadisticas del dataset: 
print(df.describe())

#Conteo de cuantas especies exactas tiene el dataset:
print(df['Species'].value_counts())

#A continuacion se mostrara la distribucion de los datos a partir de la longitud de la cetosa y del sepalo en centimetros:
df.Species=df.Species.replace('Iris-setosa',0)
df.Species=df.Species.replace('Iris-versicolor',1)
df.Species=df.Species.replace('Iris-virginica',2)

setosa     = df[df['Species'] == 0]
versicolor = df[df['Species'] ==  1]
virginica = df[df['Species'] ==  2]

plt.scatter(setosa.SepalLengthCm,
            setosa.PetalLengthCm, 
            color  ='red',
            marker ='o',
            label  ='setosa')

plt.scatter(versicolor.SepalLengthCm,
            versicolor.PetalLengthCm, 
            color  ='green',
            marker ='x',
            label  ='vesicolor')

plt.scatter(virginica.SepalLengthCm,
            virginica.PetalLengthCm, 
            color  ='blue',
            marker ='+',
            label  ='virginica')

plt.xlabel('Sepal')
plt.ylabel('petal')
plt.show()

#Creamos un dataset con los datos de entrada del dataset original : 

datos_iris=pd.DataFrame()
datos_iris['LongitudSepalo'] = df['SepalLengthCm']
datos_iris['AnchoSepalo'] = df['SepalWidthCm']
datos_iris['LongitudPetal'] = df['PetalLengthCm']
datos_iris['AnchoPetal'] = df['PetalWidthCm']

print(datos_iris)

#Introducimos una columna de entrada de 1 (Unos):

x=np.ones(150)
datos_iris.insert(0,'Columna 0',x)

print(datos_iris)

#Creamos el vector de salidas a partir de que la especie Iris-Setosa sera igual a 0 y las restantes tendrán valor 1.
#Se debe tener en cuenta que la especie Iris-Setosa asumira un valor de etiqueta 0.
#Las demas especies tales como la versicolor y la virginica asumiran el valor 1.
clase_flor=df.Species
clase_flor=clase_flor.replace(2,1)

print(clase_flor)

#Transformamos las columnas en un vector de entrada
X=np.array(datos_iris)
print(X)

#Transformamos la salida en un vector de salidas
y=np.array(clase_flor)
print(y)

#Desarrollo del algoritmo del perceptron:

class Perceptron(object):
    def __init__(self, tamaño_pesos, tasa_de_aprendizaje, iteraciones,umbral):
        self.aprendizaje = tasa_de_aprendizaje
        self.iteraciones = iteraciones
        self.pesos = np.random.uniform(0, 1, size=tamaño_pesos)
        
        self.umbral = umbral

    def predecir(self,datos,pesos,b):
        producto=producto_punto(datos,pesos)
        if producto + b > 0:
            prediccion=1
        else:
            prediccion=0
        return (prediccion)
    
    def producto_punto(self,valores,pesos): 
        suma=0
        for i in range(5):
            valor = valores[i]
            peso = pesos[i]
            producto = valor * peso
            suma = suma + producto    
        return (suma)

    def entrenar(self, entrada_datos, salidas):
        for _ in range(self.iteraciones):
            for datos,salida in zip (entrada_datos,salidas):
                valor_prediccion=predecir(datos,self.pesos,self.umbral)
                self.pesos[1] += self.aprendizaje * (salida - valor_prediccion) * datos[1]
                self.pesos[2] += self.aprendizaje * (salida - valor_prediccion) * datos[2]
                self.pesos[3] += self.aprendizaje * (salida - valor_prediccion) * datos[3]
                self.pesos[4] += self.aprendizaje * (salida - valor_prediccion) * datos[4]
                self.pesos[0] += self.aprendizaje * (salida - valor_prediccion) * datos[0]
                error = salida - valor_prediccion
                self.umbral += self.aprendizaje*error
    def get_pesos(self):
        return (self.pesos)
    
    def get_umbral(self):
        return(self.umbral)


#Dividimos los datos en entrenamiento y testeo 80% y 20%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

tamaño = 5
tasa_de_aprendizaje = 0.01
iteraciones = 1000
umbral = np.random.uniform(0, 1)

perceptron = Perceptron(tamaño, tasa_de_aprendizaje, iteraciones,umbral)

print('El primer umbral obtenido es: ',umbral,'Los primeros pesos obtenidos es:',perceptron.get_pesos())

perceptron.entrenar(X_train,y_train)

umbral_prediccion=perceptron.get_umbral()

pesos_prediccion=perceptron.get_pesos()

print('El ultimo umbral obtenido es: ',umbral_prediccion,'El ultimo pesos obtenidos es:',pesos_prediccion)
#perceptron.entrenar(X_train, y_train)

cantidad_predicciones=len(X_test)
aciertos=0
for i in range(cantidad_predicciones):
    if y_test[i] == perceptron.predecir(X_test[i],pesos_prediccion,umbral_prediccion):
        aciertos+=1
    print('Los valores de la prediccion',i,':')
    print('este es el valor predecido: ',perceptron.predecir(X_test[i],pesos_prediccion,umbral_prediccion))
    print('Este es el valor que deberia haber dado: ',y_test[i])
    print('\n')
print('\n')
porcentaje=(aciertos/cantidad_predicciones) * 100
print('El porcentaje de aciertos del modelo es: ',porcentaje)

#Un ejemplo de prediccion tomando un valor cualquiera del conjunto de datos original:

print('Los valores a predecir son: ',X[20],'el valor que deberia predecirse es: ',y[20])

valor_predecir=perceptron.predecir(X[20],pesos_prediccion,umbral_prediccion)

print('El valor predicho es: ',valor_predecir)




