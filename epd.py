#Util
import os
import numpy as np

#Imagem
import cv2
import PIL
from PIL import Image, ImageEnhance, ImageFilter


#Abrir imagens OpenCV
def carregarImg(caminho, destino, fator):
    arq = os.listdir(caminho)
    
    for count, im in enumerate(arq):
        img = cv2.imread(caminho +"\\"+ im)
        img = img.astype('float32')

        width, height, channels = img.shape
        nova = np.zeros([width + int(width * fator), height + int(height * fator), channels])

        for w in range(width):
            for h in range(height):
                nova[w + int((width * fator) / 2), h + int((height * fator)/2)] = img[w,h]
    
        cv2.imwrite(destino +"\\"+ im, nova)

corte = 0.3
#imagens originais
datasetCaminho = "D:\Projeto\Dataset\Arquivos"
#imagens resultantes
datasetDestino = "D:\\testedb\\"        
carregarImg(datasetCaminho, datasetDestino, corte)