#Util
import os
import numpy as np

#Imagem
import cv2
import PIL
from PIL import Image, ImageEnhance, ImageFilter

#CNN
import tensorflow as tf
import keras
import sklearn
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Input, Flatten
from keras.models import Model
import h5py

#opencv
#Ajustar o tamanho das imagens
def adequarImg(size, caminho, destino, expandir, corte):
    arq = os.listdir(caminho)
    box = [size*corte, size*corte, size - size*corte, size - size*corte]

    for im in arq:
        img = PIL.Image.open(caminho +"\\"+ im)
        img = img.resize((size,size), PIL.Image.ANTIALIAS)

        if expandir == True:
            im = im.replace(".bmp","")
            lista = [img, img.rotate(-45), img.rotate(45), img.rotate(-60), img.rotate(60)]
   
            for count, item in enumerate(lista):
                item = item.crop(box)
                item = item.resize((size,size), PIL.Image.ANTIALIAS)

                image = item
                image.save(destino+"\\"+im+"_"+str(count)+"0001.bmp")

                image = ImageEnhance.Brightness(item).enhance(1.7)
                image.save(destino+"\\"+im+"_"+str(count)+"0002.bmp")

                image = ImageEnhance.Brightness(item).enhance(1.5)
                image.save(destino+"\\"+im+"_"+str(count)+"0003.bmp")

                image = ImageEnhance.Brightness(item).enhance(1.3)
                image.save(destino+"\\"+im+"_"+str(count)+"0004.bmp")

                image = ImageEnhance.Brightness(item).enhance(0.7)
                image.save(destino+"\\"+im+"_"+str(count)+"0005.bmp")

                image = ImageEnhance.Brightness(item).enhance(0.5)
                image.save(destino+"\\"+im+"_"+str(count)+"0006.bmp")

                image = ImageEnhance.Brightness(item).enhance(0.2)
                image.save(destino+"\\"+im+"_"+str(count)+"0007.bmp")

                image = item.filter(ImageFilter.BLUR)
                image.save(destino+"\\"+im+"_"+str(count)+"0008.bmp")
        else:
            img = img.crop(box)
            img = img.resize((size,size), PIL.Image.ANTIALIAS)
            img = img.save(destino+"\\"+ im)


#Abrir imagens OpenCV
def carregarImg(caminho, size, cores, classes):
    arq = os.listdir(caminho)
    dataset = np.zeros([len(arq), size, size, cores])

    for count, im in enumerate(arq):
        dataset[count] = cv2.imread(caminho +"\\"+ im)

    dataset_c = np.zeros(len(arq))
    arq_classe = open(classes, 'r', encoding="utf8")
    linhas = arq_classe.read().split('\n')

    for count, linha in enumerate(linhas):
        dataset_c[count] = linha
    
    arq_classe.close()
    dataset = dataset.astype(np.uint8)

    return dataset, dataset_c

#Implementação da CNN KERAS
def cnn(x_train, y_train, x_test, y_test, size, cores, epochs, nomeArquivo):
    # since MNIST images have a single channel (not RGB, but black and white)
    # we need to include a dummy channel at the end of the definition to make this explicit
    x_train = x_train.reshape((-1, size, size, cores))
    x_test = x_test.reshape((-1, size, size, cores))

    # our input placeholder
    input_layer = Input((size, size, cores))

    # convolutional layers
    x = Conv2D(16, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = Flatten()(x)

    # mlp layers/fully connected
    x = Dense(32, activation='relu')(x)

    # output layer
    out = Dense(2, activation='softmax')(x)

    # wrap up the model
    model = Model(input_layer, out)
    model.compile(loss='binary_crossentropy', optimizer='sgd')

    # convert our labels to binary representation
    # ex.: 0 -> [1, 0, 0..., 0] and 1 -> [0, 1, 0, ..., 0]
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # train our model, do not forget to keep an eye for overfitting
    model.fit(x_train, y_train,
    batch_size=128, epochs= epochs, validation_data=(x_test, y_test))

    model.save(nomeArquivo)


def carregarCNN(arquivo):
    return keras.models.load_model(arquivo)


def main():
    #tamanho das imagens para treino
    size = 28
    corte = 0.10
    cores = 3
    #imagens originais
    datasetCaminho = "D:\Projeto\Dataset\Arquivos"
    #imagens resultantes
    datasetDestino = "D:\\Projeto\\Dataset\\Normal"
    #classes das imagens
    datasetClasses = "D:\Projeto\Dataset\classificacao.txt"
    #imagens teste
    testeCaminho = "D:\Projeto\Dataset\Teste"
    #imagens resultantes teste
    testeDestino = "D:\\Projeto\\Dataset\\NormalTeste"
    #classes das imagens teste
    testeClasses = "D:\Projeto\Dataset\classificacaoTeste.txt"
    #Salvar o modelo no caminhp
    modeloDestino = "sadCNNModel"
    #epochs
    epochs = 100

    #base treino
    adequarImg(size, datasetCaminho, datasetDestino, True, corte)
    #dataset, dataset_c = carregarImg(datasetDestino, size, cores, datasetClasses)
    
    #base teste
    #adequarImg(size, testeCaminho, testeDestino, False, corte)
    #teste, teste_c = carregarImg(testeDestino, size, cores, testeClasses)

    #cnn(dataset, dataset_c, teste, teste_c, size, cores, epochs, modeloDestino)
    #img = cv2.imread(destino +"\\pare_0001.bmp")

    #cv2.imshow('PARE', dataset[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()