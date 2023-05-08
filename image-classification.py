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

train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, (224,224), y))
train_ds = train_ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y))

train_ds = train_ds.shuffle(buffer_size=1024)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(lambda x, y: (tf.image.central_crop(x, central_fraction=0.8), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, (224,224), y))
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

class NetworkBlock(Layer):

  def __init__(self, out_channels, first_stride=1):
    super().__init__()

    first_padding = 'same'
    if first_stride != 1:
      first_padding = 'valid'
    
    self.conv_sequence = Sequential([
        Conv2D(out_channels, 3, first_stride, padding=first_padding),
        BatchNormalization(),
        ReLU(),

        Conv2D(out_channels, 3, 1, padding='same'),
        BatchNormalization(),
        ReLU()
    ])

  def call(self, inputs):
    x = self.conv_sequence(inputs)

    if x.shape == inputs.shape:
      x = x + inputs 
    
    return x

layer = NetworkBlock(4)
print(layer)

class NeuralNetwork(Model):
  def __init__(self):
    super(NeuralNetwork, self).__init__()

    self.conv_1 = Sequential([
                              Conv2D(64, 7, 2),
                              ReLU(),
                              MaxPooling2D(3, 2)
    ])

    self.resnet_chains = Sequential([NetworkBlock(64), NetworkBlock(64)] +
                                    [NetworkBlock(128, 2), NetworkBlock(128)] +
                                    [ResNetBlock(256, 2), NetworkBlock(256)] +
                                    [NetworkBlock(512, 2), NetworkBlock(512)] 
                                    )
    
    self.out = Sequential([GlobalAveragePooling2D(),
                           Dropout(0.8),
                           Dense(102, activation='softmax')])
    
  def call(self, x):
    x = self.conv_1(x)
    x = self.resnet_chains(x)
    x = self.out(x)
    return x

model = NeuralNetwork()
print(model)



model.compile(optimizer='Adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class MyCallBack(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs.get('val_accuracy') >= 0.60 :
      self.model.stop_training = True

callback = MyCallBack()

callback = CustomCallback()
model.fit(train_ds, epochs=300,validation_data = valid_ds,validation_freq =1,callbacks=[callback])

model.evaluate(test_ds)
