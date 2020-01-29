import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2


conv_layer = "block5_conv3"

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

conv_layer = model.get_layer(conv_layer)
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
cv2.imwrite("./images/elephant_hm.jpg", hm)

superimposed_image = np.uint8(hm * 0.4 + img)
cv2.imshow("Superimposed image", superimposed_image)
cv2.waitKey(0)
cv2.imwrite("./images/elephant_and_heatmap.jpg", superimposed_image)
