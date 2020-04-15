import tensorflow as tf

# for the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# base model
from tensorflow.keras.applications.vgg16 import VGG16

# for the data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# analysis
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

try:
    assert tf.__version__ >= "2.0.0"
except Exception as e:
    print(f"Error Wrong version of tensorflow")
    sys.exit()


# parameters
input_size = (224, 224, 3)
train_dir = "./train"
val_dir = "./val"
test_dir = "./test"

# hyper parameters
batch_size = 8
epochs = 30
val_split = 0.33

folders = glob(train_dir + '/*')
classes = len(folders)
image_files = glob(train_dir + '/*/*.jp*g')
val_image_files = glob(val_dir + '/*/*.jp*g')

# callbacks
checkpointer = ModelCheckpoint(filepath="catsVdogs_model_with_transfer_learning.hdf5",
                               monitor='val_acc',
                               verbose=1,
                               save_weights_only=False,
                               save_best_only=True)
early = EarlyStopping(monitor='val_acc',
                      min_delta=0,
                      patience=20,
                      verbose=1,
                      mode='auto')


def build_model(dense_units=[4096, 4096],
                drop_out=0.0,
                optimFun="RMSprop",
                lr=1e-4):
    """ build model """

    # Clear session
    tf.keras.backend.clear_session()

    # load reinforecement learning model
    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=input_size)

    for layer in base_model.layers:
        layer.trainable = False

    # Build network
    x = Flatten()(base_model.output)
    for k, v in enumerate(dense_units):
        x = Dense(units=v, activation="relu", name=f"myDense_{k}")(x)

    # output layer
    output = Dense(classes, activation="softmax", name='myPrediction')(x)

    # define model
    model = models.Model(base_model.input, output)
    print(
        f"Trainable prameters before freeze {len(model.trainable_weights)}"
    )
    print(
        f"Trainable prameters after freeze {len(model.trainable_weights)}"
    )

    if optimFun == "Adam":
        opt = Adam(learning_rate=lr)
    else:
        opt = RMSprop(learning_rate=lr)

    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["acc"])

    return model


def plot_curves(history):
    x = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(12, 10))
    plt.plot(x, history.history["acc"], label="accuracy")
    plt.plot(x, history.history["loss"], label="loss")
    try:
        plt.plot(x, history.history["val_acc"], label="validation accuracy")
        plt.plot(x, history.history["val_loss"], label="validation loss")
    except Exception as e:
        pass

    plt.ylabel("Accuracy and Loss")
    plt.xlabel("Epochs")
    plt.ylim([0, 1.1 * np.max(history.history["acc"])])
    plt.xticks(range(1, len(history.history["loss"]) + 1), x)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # set up the generators
    train_data_gen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_data_gen = ImageDataGenerator(rescale=1/255.0)

    train_gen = train_data_gen.flow_from_directory(
        train_dir,
        target_size=input_size[:2],
        shuffle=True,
        batch_size=batch_size
    )

    val_gen = train_data_gen.flow_from_directory(
        val_dir,
        target_size=input_size[:2],
        shuffle=True,
        batch_size=batch_size
    )

    # build model
    model = build_model()
    print(model.summary())
    history = model.fit_generator(
        train_gen,
        epochs=epochs,
        callbacks=[checkpointer, early],
        steps_per_epoch=len(image_files) // batch_size,
        validation_data=val_gen,
        validation_steps=len(val_image_files) // batch_size
    )
    # model.save("catsVdogs_model_with_transfer_learning.h5")
    plot_curves(history)
