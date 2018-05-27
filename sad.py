import os
import numpy as np
import cv2
import PIL
from PIL import Image as pil

#opencv
#Ajustar o tamanho das imagens
def adequarImg(size, caminho, destino):
    arq = os.listdir(caminho)

    for im in arq:
        img = PIL.Image.open(caminho +"\\"+ im)
        img = img.resize((size,size), PIL.Image.ANTIALIAS)
        img.save(destino+"\\"+ im)

#Abrir imagens OpenCV
def carregarImg(caminho, size, cores, classes):
    arq = os.listdir(caminho)
    dataset = np.zeros([len(arq),size,size,cores])

    for count, im in enumerate(arq):
        dataset[count] = cv2.imread(caminho +"\\"+ im)

    dataset_c = np.zeros(len(arq))
    arq_classe = open(classes, 'r', encoding="utf8")
    linhas = arq_classe.read().split('\n')

    for count, linha in enumerate(linhas):
        dataset_c[count] = linha
    
    arq_classe.close()

    return dataset, dataset_c

def main():
    #tamanho das imagens para treino
    size = 28
    cores = 3
    #imagens originais
    caminho = "D:\Projeto\Dataset\Arquivos"
    #imagens resultantes
    destino = "D:\\Projeto\\Dataset\\Normal"
    #classes das imagens
    classes = "D:\Projeto\Dataset\classificacao.txt"
    adequarImg(size, caminho, destino)
    
    dataset, dataset_c = carregarImg(destino, size, cores, classes)
    dataset = dataset.astype(np.uint8)
    print(dataset_c.shape)
    
    img = cv2.imread(destino +"\\pare_0001.bmp")

    cv2.imshow('PARE', dataset[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()