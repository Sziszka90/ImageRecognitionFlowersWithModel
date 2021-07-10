import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

batch_size = 32
img_height = 180
img_width = 180

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

model = tf.keras.models.load_model('my_model.h5')

sunflower_path = pathlib.Path("C:/Users/FSR2BP/Desktop/ImageRecognitionFlowers/flower_photos/flower.jpg")

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

