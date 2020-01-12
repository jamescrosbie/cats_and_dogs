# summarize feature map size for each conv layer
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
import argparse
import numpy as np
import random
import glob
import matplotlib.pyplot as plt

# parameters
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="", help="path to model")
ap.add_argument("-l", "--layer", default=0,
                help="Convolutional layer to visualise")
ap.add_argument("-i", "--index", default=0,
                help="Convolutional filter to visualise")
ap.add_argument("-e", "--epochs", default=100,
                help="Number of epochs to train model for")
args = vars(ap.parse_args())

# load the model
model = VGG16(weights="imagenet") if args["model"] == "" else models.load_model(
    args["model"])

# summarize feature map shapes
conv_layers = []
for i in range(len(model.layers)):
    layer = model.layers[i]
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    # summarize output shape
    print(i, layer.name, layer.output.shape)
    conv_layers.append([i, layer.name])

layer_index = conv_layers[int(args["layer"])][0]
layer_name = conv_layers[int(args["layer"])][1]
print(f"Using layer {layer_index} with name {layer_name}")

# #############################################
# What the feature maps see from an image
# #############################################

# define new model with output as layer of interest
model = models.Model(inputs=model.inputs,
                     outputs=model.get_layer(layer_name).output)

layer_output = model.get_layer(layer_name).output

# select a random image
files_1 = glob.glob("./test/test_cats/*")
files_2 = glob.glob("./test/test_dogs/*")
files = files_1 + files_2
i = random.randint(1, len(files))
test_image = files[i]

# load image
(h, w) = model.layers[0].output.shape[1:3]
img = image.load_img(test_image, target_size=(h, w))
plt.imshow(img)
plt.show()

# convert image to tensor
img_tensor = image.img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(f"Image shape {img_tensor.shape}")

# get feature map for layer
feature_maps = model(img_tensor)

plt.imshow(feature_maps[0, :, :, layer_index], cmap='gray')

# plot all feature maps in layer as square
square = int(np.sqrt(layer_output.shape[-1]))
ix = 1
for _ in range(square):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
        ix += 1
plt.show()

# #############################################
# What features activation maps are looking for
# #############################################


def deprocess(x):
    x -= x.mean()
    x /= x.std() + 1e-05
    x *= 0.1 * x

    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def get_Feature_map(idx):
    print(f"Deriving feature map for filter {layer_name} index {idx}")
    input_ = np.random.random((1, h, w, 3)) * 20 + 128.
    input_ = tf.Variable(tf.cast(input_, tf.float32))

    step = 1.
    for i in range(int(args["epochs"])):
        with tf.GradientTape() as tape:
            # make prediction
            layer_output = model(input_)
            # calculate loss
            loss_value = tf.reduce_mean(layer_output[:, :, :, idx])

        # calculate gradient
        grad = tape.gradient(loss_value, input_)
        grad_norm = grad / tf.sqrt(tf.reduce_mean(tf.square(grad)) + 1e-05)
        if i % 1000 == 0:
            print(f"\tIteration {i}\tLoss : {loss_value}")
        input_.assign_add(grad_norm * step)

    input_ = tf.convert_to_tensor(input_).numpy()
    img = deprocess(input_[0])
    return img

img = get_Feature_map(layer_index)
plt.imshow(img[:,:,0], cmap='gray')
plt.show()

ix = 1
for _ in range(square):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        img = get_Feature_map(ix - 1)
        plt.imshow(img[:,:,0], cmap='gray')
        ix += 1
plt.show()
