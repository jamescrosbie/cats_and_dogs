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
    if "conv" in layer.name:
        conv_layers.append((i, layer.name))
    # summarize output shape
    print(f"Layer index {i}, Name: {layer.name}, Shape: {layer.output.shape}")

layer_index = args["layer"]
layer_name = model.layers[layer_index].name
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
plt.show()

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


# show first map
img = get_Feature_map(0)
plt.imshow(img[:, :, 0])
plt.show()

# plot more feature maps
ix = 1
for _ in range(square):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        img = get_Feature_map(ix - 1)
        plt.imshow(img[:, :, 0])
        ix += 1
plt.show()


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

conv_layer = model.get_layer(conv_layers[-1][1])
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

plt.imshow(heatmap[0])
plt.show()

img = cv2.imread(image_path)
print(f"image shape {img.shape}")
print(f"heatmap shape before {heatmap.shape}")
heatmap = cv2.resize(heatmap[0], (img.shape[1], img.shape[0]))
print(f"heatmap shape after {heatmap.shape}")

heatmap = np.uint8(heatmap * 255.)
hm = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
cv2.imshow("Heatmap", hm)
cv2.waitKey(0)

superimposed_image = np.uint8(hm * 0.4 + img)
cv2.imshow("Superimposed image", superimposed_image)
cv2.waitKey(0)
