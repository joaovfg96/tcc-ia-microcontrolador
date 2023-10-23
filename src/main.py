import cv2                               # OpenCV
import numpy as np                         # NumPy

from keras.datasets import mnist         # Importando o dataset usado no treino
from keras.models import Sequential      # Modelo de rede neural
from keras.layers import Dense           # Layer do tipo densamente conectado
from keras.utils import to_categorical   # Usaremos dela o metodo 'to_categorical()'

#Inicializamos e carregamos o dataset
(imagem_treino, identificador_treino), (imagem_teste, identificador_teste) \
    = mnist.load_data()

# Reorganizamos as matrizes sem alterar os valores
imagem_treino = imagem_treino.reshape(60000, 784)
imagem_teste = imagem_teste.reshape(10000, 784)

# Aplicamos a normalização
imagem_treino = imagem_treino / 255.0
imagem_teste = imagem_teste / 255.0

#pegamos os identificadores e fazemos uma lista de matrizes binárias
identificador_treino = to_categorical(identificador_treino)
identificador_teste = to_categorical(identificador_teste)

# Criando o modelo sequencial
model = Sequential()

# Criando a camada oculta
model.add(Dense(256, activation='relu', input_dim=784))

# Criando a camada de saída
model.add(Dense(10, activation='softmax')) 

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#treinando a rede neural

model.fit(imagem_treino, identificador_treino, epochs=7)




estimador = model.evaluate(imagem_treino, identificador_treino)

print('\n Perda:{:3.2f}, Precisao:{:2.2f}'.format(estimador[0], estimador[1]))

# img = cv2.imread('numero.png')
# img = cv2.resize(img, (28, 28))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = img.reshape(1, 28*28)
# img = img/255.0

img = imagem_teste[1].reshape(1, 784)

resultado = model.predict(img)

print('Valor previsto: ',resultado.argmax())
print('Precisão: {:4.2f}%'.format(resultado.max()* 100))