#Util
import os
import numpy as np

#Imagem
import cv2
import PIL
from PIL import Image, ImageEnhance, ImageFilter

#CNN
# import tensorflow as tf
# import keras
# import sklearn
# from keras.layers import Dense, Conv2D, MaxPooling2D
# from keras.layers import Input, Flatten
# from keras.models import Model

#CNNDois
# import keras
# from keras.layers import Dense, Flatten, Activation, Dropout
# from keras.layers import Conv2D, MaxPooling2D
# from keras.models import Sequential
# from keras.optimizers import SGD
# import matplotlib.pylab as plt
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from sklearn.metrics import log_loss
from load_cifar10 import load_cifar10_data

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
            #lista = [img]
   
            for count, item in enumerate(lista):
                item = item.crop(box)
                item = item.resize((size,size), PIL.Image.ANTIALIAS)

                image = item
                image.save(destino+"\\"+im+"_"+str(count)+"0001.bmp")

                # image = ImageEnhance.Brightness(item).enhance(1.7)
                # image.save(destino+"\\"+im+"_"+str(count)+"0002.bmp")

                image = ImageEnhance.Brightness(item).enhance(1.5)
                image.save(destino+"\\"+im+"_"+str(count)+"0003.bmp")

                # image = ImageEnhance.Brightness(item).enhance(1.3)
                # image.save(destino+"\\"+im+"_"+str(count)+"0004.bmp")

                image = ImageEnhance.Brightness(item).enhance(0.7)
                image.save(destino+"\\"+im+"_"+str(count)+"0005.bmp")

                # image = ImageEnhance.Brightness(item).enhance(0.5)
                # image.save(destino+"\\"+im+"_"+str(count)+"0006.bmp")

                image = ImageEnhance.Brightness(item).enhance(0.2)
                image.save(destino+"\\"+im+"_"+str(count)+"0007.bmp")

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

    # dataset_c = np.zeros(len(arq))
    # arq_classe = open(classes, 'r', encoding="utf8")
    # linhas = arq_classe.read().split('\n')

    # for count, linha in enumerate(linhas):
    #     dataset_c[count] = linha
    
    # arq_classe.close()
    #dataset = dataset.astype(np.uint8)
    print(dataset_c)
    return dataset, dataset_c


def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    """VGG 16 Model for Keras
    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('D:\Projeto\Datase\vgg16_weights.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def cnnVGG(x_train, y_train, x_test, y_test, size, cores, epochs, nomeArquivo):
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = size, size # Resolution of inputs
    channel = cores
    num_classes = size 
    batch_size = 16 
    nb_epoch = 10

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    # Load our model
    model = vgg16_model(img_rows, img_cols, channel, num_classes)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def cnnC10(x_train, y_train, x_test, y_test, size, cores, epochs, nomeArquivo):
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
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

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

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
    size = 45
    #Salvar o modelo no caminho
    modeloDestino = "sadCNNModel"
    
    if criarRede == True:
        
        corte = 0.1
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
        #epochs
        epochs = 50

        #base treino
        adequarImg(size, datasetCaminho, datasetDestino, True, corte)
        dataset, dataset_c = carregarImg(datasetDestino, size, cores, datasetClasses)
        
        #base teste
        adequarImg(size, testeCaminho, testeDestino, False, corte)
        teste, teste_c = carregarImg(testeDestino, size, cores, testeClasses)
        
        #model = cnn(dataset, dataset_c, teste, teste_c, size, cores, epochs, modeloDestino)
        model = cnnVGG(dataset, dataset_c, teste, teste_c, size, cores, epochs, modeloDestino)

        # t = PIL.Image.open("D:\img.bmp")
        # t = t.resize((size,size), PIL.Image.ANTIALIAS)
        # inp = np.array(t).reshape((-1, size, size, cores))
        # result = model.predict(inp)
        # print(result)
    else:
        #img = cv2.imread("D:\imgP01.bmp")
        #classificar(modeloDestino, "D:\imgV80.bmp", size)
        classificar(modeloDestino, "D:\imgV40.bmp", size)
        #cv2.imshow('PARE', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()