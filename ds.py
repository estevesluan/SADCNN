#Util
import os
import numpy as np

#Imagem
import cv2
import PIL
from PIL import Image, ImageEnhance, ImageFilter

# #CNN
# import tensorflow as tf
# import keras
# import sklearn
# from keras.layers import Dense, Conv2D, MaxPooling2D
# from keras.layers import Input, Flatten
# from keras.models import Model

#CNNDois
import keras
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pylab as plt


#opencv
#Ajustar o tamanho das imagens
def adequarImg(size, caminho, destino, expandir, corte):
    arq = os.listdir(caminho)

    for item in os.listdir(destino):
        if os.path.isfile(destino +"\\"+ item):
            os.remove(destino +"\\"+ item)

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
                image.save(destino+"\\"+im+"_"+str(count)+"0001.jpg")

                # image = ImageEnhance.Brightness(item).enhance(1.7)
                # image.save(destino+"\\"+im+"_"+str(count)+"0002.bmp")

                image = ImageEnhance.Brightness(item).enhance(1.5)
                image.save(destino+"\\"+im+"_"+str(count)+"0003.jpg")

                # image = ImageEnhance.Brightness(item).enhance(1.3)
                # image.save(destino+"\\"+im+"_"+str(count)+"0004.bmp")

                image = ImageEnhance.Brightness(item).enhance(0.7)
                image.save(destino+"\\"+im+"_"+str(count)+"0005.jpg")

                # image = ImageEnhance.Brightness(item).enhance(0.5)
                # image.save(destino+"\\"+im+"_"+str(count)+"0006.bmp")

                image = ImageEnhance.Brightness(item).enhance(0.2)
                image.save(destino+"\\"+im+"_"+str(count)+"0007.jpg")

                # image = item.filter(ImageFilter.BLUR)
                # image.save(destino+"\\"+im+"_"+str(count)+"0008.bmp")
        else:
            #img = img.crop(box)
            #img = img.resize((size,size), PIL.Image.ANTIALIAS)
            img = img.save(destino+"\\"+ im)


#Abrir imagens OpenCV
def carregarImg(caminho, size, cores, classes):
    arq = os.listdir(caminho)
    dataset = np.zeros([len(arq), size, size, cores])
    dataset_c = np.zeros(len(arq))

    for count, im in enumerate(arq):
        img = cv2.imread(caminho +"\\"+ im)
        img = img.astype('float32')

        for i in range(cores):
            img[i-1] /= 255

        dataset[count] = img

        if im.find("pare") != -1:
            dataset_c[count] = 0
        if im.find("velocidade_20") != -1:
            dataset_c[count] = 1
        if im.find("velocidade_30") != -1:
            dataset_c[count] = 2
        if im.find("velocidade_40") != -1:
            dataset_c[count] = 3
        if im.find("velocidade_60") != -1:
            dataset_c[count] = 4
        if im.find("velocidade_80") != -1:
            dataset_c[count] = 5
        if im.find("velocidade_90") != -1:
            dataset_c[count] = 6
        if im.find("velocidade_100") != -1:
            dataset_c[count] = 7
        if im.find("velocidade_110") != -1:
            dataset_c[count] = 8

        # if im.find("pare") != -1:
        #     dataset_c[count] = 0
        # if im.find("velocidade_20") != -1:
        #     dataset_c[count] = 1
        # if im.find("velocidade_30") != -1:
        #     dataset_c[count] = 1
        # if im.find("velocidade_40") != -1:
        #     dataset_c[count] = 1
        # if im.find("velocidade_60") != -1:
        #     dataset_c[count] = 1
        # if im.find("velocidade_80") != -1:
        #     dataset_c[count] = 1
        # if im.find("velocidade_90") != -1:
        #     dataset_c[count] = 1
        # if im.find("velocidade_100") != -1:
        #     dataset_c[count] = 1
        # if im.find("velocidade_110") != -1:
        #     dataset_c[count] = 1

    # dataset_c = np.zeros(len(arq))
    # arq_classe = open(classes, 'r', encoding="utf8")
    # linhas = arq_classe.read().split('\n')

    # for count, linha in enumerate(linhas):
    #     dataset_c[count] = linha
    
    # arq_classe.close()
    #dataset = dataset.astype(np.uint8)
    #rint(dataset_c)
    return dataset, dataset_c

def cnnBRTSD(x_train, y_train, x_test, y_test, size, cores, epochs, nomeArquivo):
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(7))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=32,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
              
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def cnnC10(x_train, y_train, x_test, y_test, size, cores, epochs, nomeArquivo):
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=32,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
    
    model.save(nomeArquivo)
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def cnnC10P84(x_train, y_train, x_test, y_test, size, cores, epochs, nomeArquivo):
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=32,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
              
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def cnnDois(x_train, y_train, x_test, y_test, size, cores, epochs, nomeArquivo):

    x_train = x_train.reshape((-1, size, size, cores))
    x_test = x_test.reshape((-1, size, size, cores))
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    input_shape = (size, size, cores)
    num_classes = 7

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(2000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
    
    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()

    model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[history])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(range(1, 11), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

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
    out = Dense(7, activation='softmax')(x)

    # wrap up the model
    model = Model(input_layer, out)
    model.compile(loss='binary_crossentropy', optimizer='sgd')

    # convert our labels to binary representation
    # ex.: 0 -> [1, 0, 0..., 0] and 1 -> [0, 1, 0, ..., 0]
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # train our model, do not forget to keep an eye for overfitting
    model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_data=(x_test, y_test))
    model.save(nomeArquivo)
    return model

def carregarCNN(arquivo):
    return keras.models.load_model(arquivo)

def classificar(arquivoRede, arquivoImg, size, cores = 3):
    model = carregarCNN(arquivoRede)

    img = PIL.Image.open(arquivoImg)
    img = img.resize((size,size), PIL.Image.ANTIALIAS)
    inp = np.array(img).reshape((-1, size, size, cores))
    result = model.predict(inp)
    print(result)

def main():
    criarRede = True
    #criarRede = False
    #tamanho das imagens para treino
    size = 32
    #Salvar o modelo no caminho
    modeloDestino = "sadCNNModelDuasClasses"
    
    if criarRede == True:
        
        corte = 0.1
        cores = 3
        #imagens originais
        ds = "D:\Projeto\Dataset\Arquivos"
        dsAtualizado = "D:\Projeto\Dataset\ArquivosAtualizados"
        dsAtualizadoRestrito = "D:\Projeto\Dataset\ArquivosAtualizadosRestrito"
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
        #epochs
        epochs = 1

        #base treino
        adequarImg(size, dsAtualizado, datasetDestino, True, corte)
        dataset, dataset_c = carregarImg(datasetDestino, size, cores, datasetClasses)
        
        #base teste
        adequarImg(size, dsAtualizado, testeDestino, False, corte)
        teste, teste_c = carregarImg(testeDestino, size, cores, testeClasses)
        
        model = cnnC10(dataset, dataset_c, teste, teste_c, size, cores, epochs, modeloDestino)
        #model = cnnVGG(dataset, dataset_c, teste, teste_c, size, cores, epochs, modeloDestino)

        # t = PIL.Image.open("D:\img.bmp")
        # t = t.resize((size,size), PIL.Image.ANTIALIAS)
        # inp = np.array(t).reshape((-1, size, size, cores))
        # result = model.predict(inp)
        # print(result)
    else:
        #img = cv2.imread("D:\\Projeto\\Revisão Imagens\\teste.bmp")
        classificar(modeloDestino, "D:\\Projeto\\Revisão Imagens\\teste11.bmp", size)
        #classificar(modeloDestino, "D:\imgV40.bmp", size)
        #cv2.imshow('PARE', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()