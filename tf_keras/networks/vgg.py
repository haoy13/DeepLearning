import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, Model, Input

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def VGG16(inputs, nb_classes):
    output = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    output = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = layers.Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu')(output)
    output = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = layers.Conv2D(512, kernel_size=(1, 1), padding='same', activation='relu')(output)
    output = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = layers.Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = layers.Conv2D(512, kernel_size=(1, 1), padding='same', activation='relu')(output)
    output = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = layers.Flatten()(output)

    output = layers.Dense(4096, activation='relu')(output)
    output = layers.Dense(4096, activation='relu')(output)
    output = layers.Dense(nb_classes, activation='softmax')(output)

    model = Model(inputs, output)
    model.summary()
    return model




# for i in range(10):
#     model.fit(train_images[:100], train_labels[:100])
