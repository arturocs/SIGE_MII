# Ejercicio 1

Crea un cuaderno .Rmd a partir de [titanic.Rmd](https://github.com/jgromero/sige2021/blob/main/teoría/tema 3/titanic/titanic.Rmd) que realice las siguientes tareas de clasificación automática empleando **caret**:



1. Aprendizaje de un modelo de clasificación utilizando 'Random Forest' (lo llamaremos *modelo_rf1*) [ver línea 437 y siguientes]
2. Variación sobre *modelo1* utilizando una partición de datos 80-20 y validación cruzada (*modelo_rf2*)
3. Comparación y selección del mejor modelo en *modelo_rf1, modelo_rf2* términos de precisión y AUC (*modelo_rf)*
4. Aprendizaje de modelo de clasificación utilizando redes neuronales - perceptrón multicapa y parámetros por defecto (*modelo_rna*)
5. Envío de solución a Kaggle con *modelo_rna*
6. Mejora de *modelo_rna* mediante entrenamiento con rejilla de parámetros para los parámetros .size, .decay (*modelo_rna_mejorado)*
7. Envío de solución a Kaggle con *modelo_rna_mejorado*
8. Comparación de *modelo_rna_mejorado* con *modelo_rf*

Sube el fichero .Rmd con las ampliaciones, indicando al final del mismo en una tabla las métricas obtenidas para *modelo_rf* y *modelo_rna_mejorado* en entrenamiento, validación y test.



# Ejercicio 2

Crea un cuaderno .Rmd a partir de [titanic-dl.Rmd](https://github.com/jgromero/sige2021/blob/main/teoría/tema 4/titanic/titanic-dl.Rmd) para mejorar los resultados del modelo predicción del fichero:

Puedes modificar, entre otros:

1. Arquitectura de la red
2. Algoritmos de optimización 
3. Hiperparámetros (tasa de aprendizaje)
4. Otros aspectos: número de *epochs*, tamaño del *batch*, etc.

Sube el cuaderno, indicando al final del mismo en una tabla las métricas de tu modelo mejorado. 



# Ejercicio 3

Crea un cuaderno .Rmd a partir de [mnist-cnn.R](https://github.com/jgromero/sige2021/blob/main/teoría/tema 4/mnist/mnist-cnn.R) para mejorar los resultados del modelo predicción del fichero:

Puedes modificar, entre otros:



1. Arquitectura de la red
2. Algoritmos de optimización 
3. Hiperparámetros (tasa de aprendizaje)
4. Otros aspectos: número de *epochs*, tamaño del *batch*, etc.

Sube el cuaderno, indicando al final del mismo en una tabla las métricas de tu modelo mejorado.



# Ejercicio 4

Revisa el [cuaderno de Google Colaboratory](https://colab.research.google.com/drive/1ClnvFQcNo61URlg5aktHLu9LnlDgVaXV?usp=sharing) que entrena a un agente para jugar a Breakout.

Modifica el cuaderno para resolver el juego del Pong, teniendo en cuenta que:



- Las imágenes de entrada del Pong tienen el mismo tamaño que las de Breakout.
- El nombre del entorno Pong es *PongNoFrameskip-v4**.*
- El número de acciones posibles en Pong es 6, en lugar de las 4 de Breakout.
- Queremos añadir una capa convolutiva [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) llamada *layer0* adicional a la red DQN con 16 filtros, tamaño de kernel 8, strides igual a 2 y activación "relu".
- Consideramos Pong solucionado cuando la recompensa acumulada en los últimos 100 episodios (*running_reward*) es mayor que -15. 

Sube el cuaderno (fichero .ipynb) resultante. No es necesario alcanzar una  solución; probablemente, Google Colab cerrará la sesión antes de  finalizar el entrenamiento.
