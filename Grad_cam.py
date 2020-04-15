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

# assert tf.test.is_gpu_available() == True
# print(f"TensorFlow running on {tf.test.gpu_device_name}")

# parameters
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="",
                help="Model to use in visualization")
args = vars(ap.parse_args())
print(f"Viewing parameters {args}")

file_1 = glob.glob("./test/test_cats/*")
file_2 = glob.glob("./test/test_dogs/*")
images = []
images.append(file_1[random.randint(1, len(file_1))])
images.append(file_2[random.randint(1, len(file_2))])

# # load model
# model = models.load_model(
#     args["model"]) if args["model"] else VGG16(weights="imagenet")
model = models.load_model("catsVdogs_model_with_transfer_learning.hdf5")

(h, w) = model.layers[0].output.shape[1:3]

heatmap_model = models.Model(inputs=model.inputs,
                             outputs=[model.get_layer(layer_name).output, model.output])

final_dense = model.get_layer("myPrediction")
W = final_dense.get_weights()[0]

for k, image_path in enumerate(images[0]):
    # convert image to tensor
    img = image.load_img(image_path, target_size=(h, w))

    img_tensor = image.img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)

    # Get gradient of the winner class w.r.t. the output of the conv. layer
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

    img = cv2.imread(image_path)
    cv2.imshow("Image", img)
    cv2.imwrite(f"./images/img2_{k}.jpg", img)
    print(f"image shape {img.shape}")
    print(f"heatmap shape before {heatmap.shape}")
    heatmap = cv2.resize(heatmap[0], (img.shape[1], img.shape[0]))
    print(f"heatmap shape after {heatmap.shape}")

    heatmap = np.uint8(heatmap * 255.)
    hm = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imshow("Heatmap", hm)
    cv2.waitKey(0)
    cv2.imwrite(f"./images/heatmap2_{k}.jpg", hm)

    superimposed_image = np.uint8(hm * 0.4 + img)
    cv2.imshow("Superimposed image", superimposed_image)
    cv2.waitKey(0)
    cv2.imwrite(f"./images/superimposed2_{k}.jpg", superimposed_image)
