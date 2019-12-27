import tensorflow as tf

# for the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop

# base model
from tensorflow.keras.applications import VGG16

# for the data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# analysis
import sys
import numpy as np
import matplotlib.pyplot as plt


try:
    assert tf.__version__ == "2.0.0"
except Exception as e:
    print(f"Error Wrong version of tensorflow")
    sys.exit()


# parameters
input_size = (150, 150, 3)
train_dir = "./train"
val_dir = "./val"
test_dir = "./test"

# hyper parameters
batch_size = 32
epochs = 100
val_split = 0.33


def build_model(dense_units=[512],
                drop_out=0.0,
                optimFun="RMSprop",
                lr=1e-4):
    """ build model """

    # Clear session
    tf.keras.backend.clear_session()

    # load reinforecement learning model
    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=input_size)

    # Build network
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    for _, v in enumerate(dense_units):
        model.add(Dense(units=v, activation="relu"))

    # output layer
    model.add(Dense(1, activation="sigmoid"))

    print(
        f"Trainable prameters before freeze {len(model.trainable_weights)}"
    )
    base_model.trainable = False
    print(
        f"Trainable prameters after freeze {len(model.trainable_weights)}"
    )

    if optimFun == "Adam":
        opt = Adam(learning_rate=lr)
    else:
        opt = RMSprop(learning_rate=lr)

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc"])

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
        batch_size=batch_size,
        class_mode="binary"
    )

    val_gen = train_data_gen.flow_from_directory(
        val_dir,
        target_size=input_size[:2],
        batch_size=batch_size,
        class_mode="binary"
    )

    # build model
    model = build_model()
    print(model.summary())
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=30,
        validation_data=val_gen,
        validation_steps=50
    )
    model.save("catsVdogs_model_with_transfer_learning.h5")
    plot_curves(history)
