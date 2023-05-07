import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
# Load the dataset
train_data, test_data ,valid_data= tfds.load(name='oxford_flowers102', split=['train', 'test','validation'], as_supervised=True)

# Define the image size and batch size
img_size = (224, 224)
batch_size = 32

# Define the data augmentation pipeline
train_data = train_data.map(lambda x, y: (tf.image.central_crop(x, central_fraction=0.8), y))

train_data = train_data.map(lambda x, y: (tf.image.resize(x, img_size), y))
train_data = train_data.map(lambda x, y: (tf.image.random_flip_left_right(x), y))

train_data = train_data.shuffle(buffer_size=1024)
train_data = train_data.batch(batch_size)
train_data = train_data.prefetch(tf.data.AUTOTUNE)

test_data = test_data.map(lambda x, y: (tf.image.central_crop(x, central_fraction=0.8), y))

test_data = test_data.map(lambda x, y: (tf.image.resize(x, img_size), y))
test_data = test_data.batch(batch_size)
test_data = test_data.prefetch(tf.data.AUTOTUNE)

valid_data = valid_data.map(lambda x, y: (tf.image.central_crop(x, central_fraction=0.8), y))
valid_data = valid_data.map(lambda x, y: (tf.image.resize(x, (224,224)), y))
valid_data = valid_data.batch(batch_size)
valid_data = valid_data.prefetch(tf.data.AUTOTUNE)

class ResNetBlock(Layer):

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
      x = x + inputs # Skip connection
    
    return x

layer = ResNetBlock(4)
print(layer)

class ResNet(Model):
  def __init__(self):
    super(ResNet, self).__init__()

    self.conv_1 = Sequential([
                              Conv2D(64, 7, 2),
                              ReLU(),
                              MaxPooling2D(3, 2)
    ])

    self.resnet_chains = Sequential([ResNetBlock(64), ResNetBlock(64)] +
                                    [ResNetBlock(128, 2), ResNetBlock(128)] +
                                    [ResNetBlock(256, 2), ResNetBlock(256)] +
                                    [ResNetBlock(512, 2), ResNetBlock(512)] 
                                    )
    
    self.out = Sequential([GlobalAveragePooling2D(),
                           Dropout(0.8),
                           Dense(102, activation='softmax')])
    
  def call(self, x):
    x = self.conv_1(x)
    x = self.resnet_chains(x)
    x = self.out(x)
    return x

model = ResNet()
print(model)



model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class CustomCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs.get('val_accuracy') >= 0.53 :
      self.model.stop_training = True

callback = CustomCallback()

callback = CustomCallback()
model.fit(train_data, epochs=300,validation_data = valid_data,validation_freq =1,callbacks=[callback])

model.evaluate(test_data)
