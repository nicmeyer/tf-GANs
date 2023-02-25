from PIL import Image
from PIL import ImageDraw

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import callbacks
from tensorflow.keras import datasets
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import preprocessing

IMG_ROWS = 28
IMG_COLS = 28
CHANNELS = 1
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)
EPOCHS = 10
BATCH_SIZE = 100
LATENT_DIM = 100


class Maxout(layers.Layer):
  def __init__(self, units, pieces, initializer):
    super().__init__()
    self.units = units
    self.pieces = pieces
    self.detector_layer_dim = units * pieces
    self.initializer = initializer

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[1], self.detector_layer_dim),
                             initializer=self.initializer,
                             trainable=True)

  def call(self, x):
    x = tf.matmul(x, self.w)
    x = tf.reshape(x, (-1, self.pieces, self.units))
    return tf.reduce_max(x, 1)


def schedule(epoch, initial_learning_rate, decay_factor, min_lr):
  lr = initial_learning_rate / (decay_factor ** epoch)
  if lr < min_lr:
    return min_lr
  return lr


class ExponentialDecay(callbacks.Callback):
  def __init__(self, schedule, initial_learning_rate, decay_factor, min_lr):
    super().__init__()
    self.schedule = schedule
    self.initial_learning_rate = initial_learning_rate
    self.decay_factor = decay_factor
    self.min_lr = min_lr

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, "lr"):
      raise ValueError('Optimizer must have a "lr" attribute.')
    scheduled_lr = self.schedule(
      epoch, self.initial_learning_rate, self.decay_factor, self.min_lr)
    backend.set_value(self.model.optimizer.lr, scheduled_lr)
    print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


def build_descriminator():
  discriminator = keras.Sequential(
    [
      layers.Flatten(input_shape=IMG_SHAPE),
      layers.Dropout(.8),
      Maxout(units=240, pieces=5,
             initializer=initializers.RandomUniform(-.005, .005)),
      layers.Dropout(.5),
      Maxout(units=240, pieces=5,
             initializer=initializers.RandomUniform(-.005, .005)),
      layers.Dropout(.5),
      layers.Dense(1, activation="sigmoid",
                   kernel_initializer=initializers.RandomUniform(-.005, .005))
    ],
    name="discriminator",
  )
  return discriminator


class InitSigmoidBiasFromMarginals(initializers.Initializer):
  def __init__(self, design_matrix):
    if tf.experimental.numpy.ndim(design_matrix) != 2:
      raise ValueError("Expected design matrix to have two dimensions.")
    self.design_matrix = design_matrix

  def __call__(self, shape, dtype=None):
    x = tf.reduce_mean(self.design_matrix, 0)
    x = tf.clip_by_value(x, 1e-7, 1. - 1e-7)
    x = tf.math.divide(x, 1. - x)
    x = tf.math.log(x)
    x = tf.reshape(x, shape)
    x = tf.cast(x, dtype)
    return x


def build_generator(design_matrix):
  generator = keras.Sequential(
    [
      keras.Input(shape=(LATENT_DIM,)),
      layers.Dense(1200,
                   activation="relu",
                   kernel_initializer=initializers.RandomUniform(-.05, .05)),
      layers.Dense(1200,
                   activation="relu",
                   kernel_initializer=initializers.RandomUniform(-.005, .005)),
      layers.Dense(
        784,
        activation="sigmoid",
        bias_initializer=InitSigmoidBiasFromMarginals(design_matrix)
      ),
      layers.Reshape(IMG_SHAPE)
    ],
    name="generator",
  )
  return generator


class GAN(keras.Model):
  def __init__(self, discriminator, generator, latent_dim):
    super().__init__()
    self.discriminator = discriminator
    self.generator = generator
    self.latent_dim = latent_dim

  def compile(self, d_optimizer, g_optimizer, loss_fn):
    super().compile()
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer
    self.loss_fn = loss_fn
    self.d_loss_metric = keras.metrics.Mean(name="d_loss")
    self.g_loss_metric = keras.metrics.Mean(name="g_loss")

  @property
  def metrics(self):
    return [self.d_loss_metric, self.g_loss_metric]

  def train_step(self, real_images):
    batch_size = tf.shape(real_images)[0]
    random_latent_vectors = tf.random.normal(
      shape=(batch_size, self.latent_dim))

    generated_images = self.generator(random_latent_vectors)

    combined_images = tf.concat([generated_images, real_images], axis=0)

    labels = tf.concat(
      [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
    )

    with tf.GradientTape() as tape:
      predictions = self.discriminator(combined_images)
      d_loss = self.loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(
      zip(grads, self.discriminator.trainable_weights)
    )

    random_latent_vectors = tf.random.normal(
      shape=(batch_size, self.latent_dim))

    misleading_labels = tf.zeros((batch_size, 1))

    with tf.GradientTape() as tape:
      predictions = self.discriminator(self.generator(random_latent_vectors))
      g_loss = self.loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, self.generator.trainable_weights)
    self.g_optimizer.apply_gradients(
      zip(grads, self.generator.trainable_weights))

    self.d_loss_metric.update_state(d_loss)
    self.g_loss_metric.update_state(g_loss)
    return {
      "d_loss": self.d_loss_metric.result(),
      "g_loss": self.g_loss_metric.result(),
    }


class GANMonitor(callbacks.Callback):
  def __init__(self, rows=5, cols=5, latent_dim=100):
    self.rows = rows
    self.cols = cols
    self.latent_dim = latent_dim
    self.grids = []

  def _generate_image_grid(self):
    random_latent_vectors = tf.random.normal(
      shape=(self.rows * self.cols, self.latent_dim)
    )
    generated_images = self.model.generator(random_latent_vectors)
    generated_images *= 255
    generated_images.numpy()

    generated_images = map(preprocessing.image.array_to_img, generated_images)

    image_grid = Image.new(
      mode="RGB", size=(self.cols * IMG_COLS, self.rows * IMG_ROWS)
    )
    for i, img in enumerate(generated_images):
      image_grid.paste(
        img, box=(i % self.cols * IMG_COLS, i // self.cols * IMG_ROWS)
      )
    image_grid = image_grid.resize((500, 500))
    return image_grid

  def on_epoch_begin(self, epoch, logs=None):
    grid = self._generate_image_grid()
    self.grids.append(grid)

  def on_train_end(self, logs=None):
    grid = self._generate_image_grid()
    grid.save("gan/gan.gif", save_all=True,
              append_images=self.grids, duration=400, loop=0)


if __name__ == "__main__":
  (x_train, _), (_, _) = datasets.mnist.load_data()

  x_train = x_train / 255.
  x_train = np.expand_dims(x_train, axis=3)

  design_matrix = tf.reshape(x_train, (-1, IMG_ROWS * IMG_COLS))

  discriminator = build_descriminator()
  generator = build_generator(design_matrix)

  discriminator.summary()
  generator.summary()

  gan = GAN(discriminator=discriminator, generator=generator,
            latent_dim=LATENT_DIM)
  gan.compile(
    d_optimizer=keras.optimizers.SGD(.1, .5),
    g_optimizer=keras.optimizers.SGD(.1, .5),
    loss_fn=keras.losses.BinaryCrossentropy(),
  )

  schedule_callback = ExponentialDecay(schedule, initial_learning_rate=.1,
                                       decay_factor=1.000004, min_lr=.000001)
  gan.fit(
    x_train,
    epochs=EPOCHS,
    callbacks=[
      GANMonitor(rows=5, cols=5, latent_dim=LATENT_DIM), schedule_callback
    ]
  )
