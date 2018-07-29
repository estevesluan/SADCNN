#Util
import numpy as np
import time

#Imagem
import cv2
import PIL
from PIL import Image, ImageEnhance, ImageFilter

#CNN
import keras
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pylab as plt

def classificar(arquivoRede, arquivoImg, size, cores = 3):
    model = keras.models.load_model(arquivoRede)

    img = PIL.Image.open(arquivoImg)
    img = img.resize((size,size), PIL.Image.ANTIALIAS)
    inp = np.array(img).reshape((-1, size, size, cores))
    result = model.predict(inp)
    print(result)

def classificarCV(arquivoRede, arquivoImg, size, cores = 3):
    model = keras.models.load_model(arquivoRede)

    img = cv2.imread(arquivoImg)
    img = cv2.resize(img,(size,size))
    inp = np.array(img).reshape((-1, size, size, cores))
    start = time.time()
    result = model.predict(inp)
    print(str(time.time()-start))
    print(result)
    print('----------------------------------')

def main():
    #tamanho das imagens para treino
    size = 32
    #Salvar o modelo no caminho
    modeloDestino = "sadCNNModelDuasClasses"

    img = cv2.imread(r"H:\Arquivo de Teste\pare_001.bmp")
    img2 = cv2.imread(r"H:\Arquivo de Teste\pare_002.bmp")
    img3 = cv2.imread(r"H:\Arquivo de Teste\velocidade_001.bmp")
    img4 = cv2.imread(r"H:\Arquivo de Teste\velocidade_002.bmp")
    img5 = cv2.imread(r"H:\Arquivo de Teste\erro.bmp")
    
    #classificarCV(modeloDestino, r"C:\Users\luanesteves\Documents\Projeto\Teste\teste11.bmp", size)
    
    classificarCV(modeloDestino, r"H:\Arquivo de Teste\pare_001.bmp", size)
    classificarCV(modeloDestino, r"H:\Arquivo de Teste\pare_002.bmp", size)
    classificarCV(modeloDestino, r"H:\Arquivo de Teste\velocidade_001.bmp", size)
    classificarCV(modeloDestino, r"H:\Arquivo de Teste\velocidade_002.bmp", size)
    classificarCV(modeloDestino, r"H:\Arquivo de Teste\erro.bmp", size)
    #classificar(modeloDestino, "D:\imgV40.bmp", size)
    cv2.imshow('01', img)
    cv2.waitKey(0)
    cv2.imshow('02', img2)
    cv2.waitKey(0)
    cv2.imshow('03', img3)
    cv2.waitKey(0)
    cv2.imshow('04', img4)
    cv2.waitKey(0)
    cv2.imshow('05', img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()