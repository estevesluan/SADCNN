#Util
import numpy as np
import time

#Imagem
import cv2

#CNN
import keras

def classificar(model, img, size, cores = 3):
    img = cv2.resize(img,(size,size))
    inp = np.array(img).reshape((-1, size, size, cores))
    
    start = time.time()
    result = model.predict(inp)
    print(str(time.time()-start))
    print(result)

def main():
    #tamanho das imagens para treino
    size = 32
    #Salvar o modelo no caminho
    modelo = "sadCNNModelDuasClasses"

    img = cv2.imread("pare_001.bmp")
    img2 = cv2.imread("pare_002.bmp")
    img3 = cv2.imread("velocidade_001.bmp")
    img4 = cv2.imread("velocidade_002.bmp")
    img5 = cv2.imread("erro.bmp")
    
    #classificarCV(modeloDestino, r"C:\Users\luanesteves\Documents\Projeto\Teste\teste11.bmp", size)
    start = time.time()
    print("Carregar Modelo")
    model = keras.models.load_model(modelo)
    print(str(time.time()-start))
    
    print("Classificar")
    classificar(model, img, size)
    classificar(model, img2, size)
    classificar(model, img3, size)
    classificar(model, img4, size)
    classificar(model, img5, size)
	
    #classificar(modeloDestino, "D:\imgV40.bmp", size)
    cv2.imshow('Imagem', img)
    cv2.waitKey(0)
    cv2.imshow('Imagem', img2)
    cv2.waitKey(0)
    cv2.imshow('Imagem', img3)
    cv2.waitKey(0)
    cv2.imshow('Imagem', img4)
    cv2.waitKey(0)
    cv2.imshow('Imagem', img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()