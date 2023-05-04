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

model=keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(15,15), strides=(4,4), activation='relu', input_shape=(224,224,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(102,activation='softmax')  
    
    
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='SGD',
    metrics=['accuracy'],
    
)
model.summary()


history = model.fit(train_batches,
                    epochs=100,
                    validation_data=validation_batches,
                    validation_freq=1,
                   )

f,ax=plt.subplots(2,1,figsize=(10,10)) 

#Assigning the first subplot to graph training loss and validation loss
ax[0].plot(model.history.history['loss'],color='b',label='Training Loss')
ax[0].plot(model.history.history['val_loss'],color='r',label='Validation Loss')

#Plotting the training accuracy and validation accuracy
ax[1].plot(model.history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(model.history.history['val_accuracy'],color='r',label='Validation Accuracy')

plt.legend()

model.evaluate(test_batches)
