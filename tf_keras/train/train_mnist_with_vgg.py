import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input

from tf_keras.networks import vgg
from tf_keras.data_processor import mnist

EPOCHS = 20
BATCH_SIZE = 32
IS_SAVE_MODEL = True
SAVE_DIR = r'D:\all_workspaces\AI\DeepLearning\tf_keras\models'
SAVE_NAME = 'MNIST_VGG.h5'


def train(model, data, is_save=IS_SAVE_MODEL, save_dir=SAVE_DIR, save_name=SAVE_NAME, **kwargs):
    assert ('epochs' in kwargs and 'batch_size' in kwargs), 'epochs or batch_size not set'
    assert len(data) == 4, 'data should contains train_images, train_labels, test_images, test_labels'
    train_images, train_labels, test_images, test_labels = data
    print(test_images.shape)

    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1 / 255.

    )
    train_generator = train_gen.flow(train_images, train_labels, batch_size=kwargs['batch_size'])

    test_gen = ImageDataGenerator(
        rescale=1 / 255.
    )
    test_generator = test_gen.flow(test_images, test_labels, batch_size=kwargs['batch_size'])

    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(train_generator, validation_data=test_generator, epochs=kwargs['epochs'],
              steps_per_epoch=kwargs['batch_size'])


if __name__ == '__main__':
    model = vgg.VGG16(Input((28, 28, 1)), 10)
    train_images = mnist.load_train_images()
    train_labels = mnist.load_train_labels()
    test_images = mnist.load_test_images()
    test_labels = mnist.load_test_labels()

    train_images = np.expand_dims(train_images, axis=3)
    train_labels = tf.one_hot(train_labels, 10)
    test_images = np.expand_dims(test_images, axis=3)
    test_labels = tf.one_hot(test_labels, 10)

    train_data = (train_images, train_labels, test_images, test_labels)
    #
    train(model, train_data, epochs=EPOCHS, batch_size=BATCH_SIZE)
