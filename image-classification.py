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

class NetworkBlockOne(Layer):

  def __init__(self, filters):
    super().__init__()

   
    
    self.convDouble = Sequential([
        Conv2D(filters, 3, 2, padding='valid'),BatchNormalization(),ReLU(),

        Conv2D(filters, 3, 1, padding='same'),BatchNormalization(),ReLU()
    ])

  def call(self, params):
    x = self.convDouble(params)

    if x.shape == params.shape:
      x = x + params 
    
    return x

class NetworkBlockTwo(Layer):

  def __init__(self, filters):
    super().__init__()

   
    
    self.convDouble = Sequential([
        Conv2D(filters, 3, 1, padding='same'),BatchNormalization(),ReLU(),

        Conv2D(filters, 3, 1, padding='same'),BatchNormalization(),ReLU()
        ])

  def call(self, params):
    x = self.convDouble(params)

    if x.shape == params.shape:
      x = x + params 
    
    return x

class NeuralNetwork(Model):
  def __init__(self):
    super(NeuralNetwork, self).__init__()

    self.start = Sequential([Conv2D(64, 7, 2),ReLU(),MaxPooling2D(3, 2)])

    self.resnetMiddle = Sequential([NetworkBlockOne(64), NetworkBlockTwo(64)] +
                                    [NetworkBlockOne(128), NetworkBlockTwo(128)] +
                                    [NetworkBlockOne(256), NetworkBlockTwo(256)] +
                                    [NetworkBlockOne(512), NetworkBlockTwo(512)] 
                                    )
    
    self.end = Sequential([GlobalAveragePooling2D(),Dropout(0.8),Dense(102, activation='softmax')])
    
  def call(self, x):
    x = self.start(x)
    x = self.resnetMiddle(x)
    x = self.end(x)
    return x

model = NeuralNetwork()



model.compile(optimizer='Adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class MyCallBack(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs.get('val_accuracy') >= 0.60 :
      self.model.stop_training = True

callback = MyCallBack()

model.fit(train_ds, epochs=300,validation_data = valid_ds,validation_freq =1,callbacks=[callback])

model.evaluate(test_ds)
