# summarize feature map size for each conv layer
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import argparse
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import cv2

assert tf.test.is_gpu_available() == True
print(f"TensorFlow running on {tf.test.gpu_device_name}")

# parameters
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="", help="path to model")
ap.add_argument("-l", "--layer", default=1, type=int,
                help="Convolutional layer to visualise")
ap.add_argument("-i", "--index", default=0, type=int,
                help="Convolutional filter to visualise")
ap.add_argument("-e", "--epochs", default=200, type=int,
                help="Number of epochs to train model for")
args = vars(ap.parse_args())
print(f"Viewing parameters {args}")

# load the model
model = VGG16(weights="imagenet", include_top=True) if args["model"] == "" else models.load_model(
    args["model"])


# summarize feature map shapes
conv_layers = []
for i in range(len(model.layers)):
    layer = model.layers[i]
    # summarize output shape
    print(f"Layer index {i}, Name: {layer.name}, Shape: {layer.output.shape}")
    conv_layers.append([i, layer.name])

layer_index = conv_layers[args["layer"]][0]
layer_name = conv_layers[args["layer"]][1]
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

plt.imshow(feature_maps[0, :, :, layer_index])

# plot all feature maps in layer as square
square = int(np.sqrt(layer_output.shape[-1]))
if square > 8:
    square = 8
ix = 1
for _ in range(square):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(feature_maps[0, :, :, ix - 1])
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
    print(f"Deriving feature map for layer {layer_name} index {idx}")
    input_ = np.random.random((1, h, w, 3)) * 20 + 128.
    input_ = tf.Variable(tf.cast(input_, tf.float32))

    step = 1.
    for i in range(args["epochs"]):
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
plt.imshow(img[:, :, 0])
plt.show()

# square = 8
# ix = 1
# for _ in range(square):
#     for _ in range(square):
#         # specify subplot and turn of axis
#         ax = plt.subplot(square, square, ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # plot filter channel in grayscale
#         img = get_Feature_map(ix - 1)
#         plt.imshow(img[:, :, 0])
#         ix += 1
# plt.show()


# #############################################
# Heat maps
# #############################################

# convert image to tensor
image_path = "./images/elephant.jpg"
img = image.load_img(image_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = preprocess_input(img_tensor)

# make a prediction
model = VGG16(weights="imagenet")
pred = model.predict(img_tensor)
print(f"Predition {decode_predictions(pred, top=3)[0]}")

last_conv_layer = model.get_layer("block5_conv3")
heatmap_model = models.Model(
    [model.inputs], [last_conv_layer.output, model.output])

with tf.GradientTape() as tape:
    # make prediction
    conv_output, predictions = heatmap_model(img_tensor)
    # calculate loss
    loss_value = predictions[:, np.argmax(predictions[0])]

# calculate gradient
grad = tape.gradient(loss_value, conv_output)
grad_norm = tf.reduce_mean(grad, axis=(0, 1, 2))
heatmap = tf.reduce_mean(tf.multiply(grad_norm, conv_output), axis=-1)

conv_layer = model.get_layer("block5_conv3")
heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

# Get gradient of the winner class w.r.t. the output of the (last) conv. layer
with tf.GradientTape() as tape:
    conv_output, predictions = heatmap_model(img_tensor)
    loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap /= max_heat

print(heatmap[0].shape)
plt.imshow(heatmap[0])
plt.show()

img = cv2.imread(image_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(heatmap * 255)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_image = heatmap * 0.4 + img
cv2.imshow(superimposed_image)
