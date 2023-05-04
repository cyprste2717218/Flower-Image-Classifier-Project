import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import tensorflow_hub as hub
import tensorflow_datasets as tfds

import pathlib 

dataset, info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)

# Create a training set, a validation set and a test set.
test_ds, train_ds, validate_ds = dataset['test'], dataset['train'], dataset['validation']

num_train = len(train_ds)
num_val = len(validate_ds)
num_test =len(test_ds)
num_classes = info.features['label'].num_classes


    
IMAGE_RES = 224
def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches = train_ds.cache().shuffle(len(train_ds)//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validate_ds.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = test_ds.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

