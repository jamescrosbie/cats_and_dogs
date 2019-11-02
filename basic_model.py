import tensorflow as tf

# for the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_accuracy

# for the data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# for the training
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
import numpy as np
import matplotlib.pyplot as plt


try:
    assert tf.__version__ == "2.0.0-beta1"
except Exception as e:
    print(f"Error Wrong version of tensorflow")
    sys.exit()


# parameters
input_size = (150, 150, 3)
train_dir = "./train"
val_dir = "./val"
test_dir = "./test"

# hypter parameters
lr = 10e-4
batch_size = 128
filters = [32, 64, 128, 512]
filter_size = [(3, 3), (3, 3), (3, 3), (3, 3)]
pooling_size = [(2, 2), (2, 2), (2, 2), (2, 2)]
dense_units = [128, 512]
drop_out = 0.0
epochs = 10
val_split = 0.33

# set callbacks
callback_ES = EarlyStopping(monitor="val_loss", patience=3)
callback_CP = ModelCheckpoint(
    filepath="./checkpoints",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    period=10,
)


def build_model():
    """ build model """

    # Clear session
    tf.keras.backend.clear_session()

    # Build network
    model = Sequential()
    # CNN
    for k, v in enumerate(filters):
        if k == 0:
            model.add(
                Conv2D(
                    filters=v,
                    kernel_size=filter_size[k],
                    padding="same",
                    activation="relu",
                    input_shape=input_size,
                )
            )
        else:
            model.add(
                Conv2D(
                    filters=k,
                    kernel_size=filter_size[k],
                    padding="same",
                    activation="relu",
                )
            )

        model.add(MaxPool2D(pool_size=pooling_size[k], strides=2, padding="valid"))

    # Dense layers
    model.add(Flatten())
    for _, v in enumerate(dense_units):
        model.add(Dense(units=v, activation="relu"))
        model.add(Dropout(drop_out))

    # output layer
    model.add(Dense(1, activation="sigmoid"))

    opt = Adam(learning_rate=lr)
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
    train_data_gen = ImageDataGenerator(rescale=1 / 255.0)
    val_data_gen = ImageDataGenerator(rescale=1 / 255.0)

    train_gen = train_data_gen.flow_from_directory(
        train_dir,
        target_size=input_size[:2],
        batch_size=batch_size,
        class_mode="binary",
    )

    val_gen = train_data_gen.flow_from_directory(
        val_dir, target_size=input_size[:2], batch_size=batch_size, class_mode="binary",
    )

    # build model
    model = build_model()
    print(model.summary())
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=50,
        callbacks=[callback_ES, callback_CP],
    )
    model.save("catsVdogs_model1.h5")
    plot_curves(history)
