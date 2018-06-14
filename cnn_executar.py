#Util
import numpy as np

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

def main():
    #tamanho das imagens para treino
    size = 32
    #Salvar o modelo no caminho
    modeloDestino = "sadCNNModelDuasClasses"

    #img = cv2.imread("D:\\Projeto\\Revisão Imagens\\teste.bmp")
    classificar(modeloDestino, r"C:\Users\luanesteves\Documents\Projeto\Teste\teste.bmp", size)
    #classificar(modeloDestino, "D:\imgV40.bmp", size)
    #cv2.imshow('PARE', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()