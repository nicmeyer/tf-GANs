import numpy as np
import tensorflow as tf
import torch

# Check how to map Keras implementation of Conv2DTranspose to Torch
# implementation of ConvTranspose2d.
r = np.random.normal(size=(20, 2, 50, 100))

tf_input = tf.convert_to_tensor(r)
tf_conv = tf.keras.layers.Conv2DTranspose(
  3,
  4,
  data_format="channels_first",
  kernel_initializer=tf.keras.initializers.Ones(),
  use_bias=False
)
tf_result = tf_conv(tf_input).numpy()

torch_input = torch.Tensor(r)

torch_conv = torch.nn.ConvTranspose2d(2, 3, 4, bias=False)
torch_conv.weight.data.fill_(1.)

torch_result = torch_conv(torch_input).detach().numpy()

assert tf_result.shape == torch_result.shape
assert np.isclose(tf_result, torch_result, atol=1e-05).all()
