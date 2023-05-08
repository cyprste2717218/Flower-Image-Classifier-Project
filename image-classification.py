import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
# Load the dataset
train_ds, test_ds ,valid_ds= tfds.load(name='oxford_flowers102', split=['train', 'test','validation'], as_supervised=True)

# Define the image size and batch size
batch_size = 32

# Define the data augmentation pipeline
train_ds = train_ds.map(lambda x, y: (tf.image.central_crop(x, central_fraction=0.8), y))

train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, (224,224)), y))
train_ds = train_ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y))

train_ds = train_ds.shuffle(buffer_size=1024)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(lambda x, y: (tf.image.central_crop(x, central_fraction=0.8), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, (224,224)), y))
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

valid_ds = valid_ds.map(lambda x, y: (tf.image.central_crop(x, central_fraction=0.8), y))
valid_ds = valid_ds.map(lambda x, y: (tf.image.resize(x, (224,224)), y))
valid_ds = valid_ds.batch(batch_size)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)


tf.keras.optimizers.Adamax(
    learning_rate=0.005,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name="Adamax",

)

model = Sequential([
    Conv2D(64,7,2),
    ReLU(),
    MaxPooling2D(3,2),

    Conv2D(64,3,1,'same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(64,3,1,'same'),
    BatchNormalization(),
    ReLU(),

    Conv2D(64,3,1,'same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(64,3,1,'same'),
    BatchNormalization(),
    ReLU(),

    Conv2D(128,3,2,'valid'),
    BatchNormalization(),
    ReLU(),
    Conv2D(128,3,1,'same'),
    BatchNormalization(),
    ReLU(),

    Conv2D(128,3,1,'same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(128,3,1,'same'),
    BatchNormalization(),
    ReLU(),

    Conv2D(256,3,2,'valid'),
    BatchNormalization(),
    ReLU(),
    Conv2D(256,3,1,'same'),
    BatchNormalization(),
    ReLU(),

    Conv2D(256,3,1,'same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(256,3,1,'same'),
    BatchNormalization(),
    ReLU(),

    Conv2D(512,3,2,'valid'),
    BatchNormalization(),
    ReLU(),
    Conv2D(512,3,1,'same'),
    BatchNormalization(),
    ReLU(),

    Conv2D(512,3,1,'same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(512,3,1,'same'),
    BatchNormalization(),
    ReLU(),

    GlobalAveragePooling2D(),
    Dropout(0.8),
    Dense(102, activation='softmax')
])






model.compile(optimizer='Adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class MyCallBack(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs.get('val_accuracy') >= 0.60 :
      self.model.stop_training = True

callback = MyCallBack()

model.fit(train_ds, epochs=300,validation_data = valid_ds,validation_freq =1,callbacks=[callback])

model.evaluate(test_ds)
