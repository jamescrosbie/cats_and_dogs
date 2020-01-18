import tensorflow as tf

# for the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop

# for the data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# for the training
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# analysis
import sys
import numpy as np
import matplotlib.pyplot as plt


try:
    assert tf.__version__ >= "2.0.0"
except Exception as e:
    print(f"Error Wrong version of tensorflow", e)
    sys.exit()


# parameters
input_size = (150, 150, 3)
train_dir = "./train/TFRecord"
val_dir = "./val/TFRecord"
test_dir = "./test/TFRecord"

# hypter parameters
batch_size = 32
epochs = 100
val_split = 0.33

# set callbacks
callback_ES = EarlyStopping(monitor="val_loss", patience=3)
callback_CP = ModelCheckpoint(
    filepath="./checkpoints",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto"
)


def load_data():
    pass


def model_metrics(y_pred, y_true):
    """Returns list of metrics to monior in the model"""
    pass


def model_loss():
    pass


def build_model(filters=[32, 64, 128, 128],
                filter_size=[(3, 3)] * 4,
                pooling_size=[(2, 2)] * 4,
                dense_units=[512],
                drop_out=0.0,
                BN=False,
                optimFun="RMSprop",
                lr=1e-4):
    """ build model """

    # Clear session
    tf.keras.backend.clear_session()

    # Build network
    model = Sequential()
    model.add(Input(shape=input_size))
    # CNN
    for k, v in enumerate(filters):
        model.add(Conv2D(filters=v, kernel_size=filter_size[k]))
        if BN:
            model.add(BatchNormalization())
        model.add(Activation("relu"))
        if pooling_size:
            model.add(
                MaxPool2D(pool_size=pooling_size[k],
                          strides=2, padding="valid")
            )

    # Dense layers
    model.add(Flatten())
    model.add(Dropout(drop_out))
    for _, v in enumerate(dense_units):
        model.add(Dense(units=v))
        model.add(Activation("relu"))
        if BN:
            model.add(BatchNormalization())

    # output layer
    model.add(Dense(1, activation="sigmoid"))

    if optimFun == "Adam":
        opt = Adam(learning_rate=lr)
    if optimFun == "RMSprop":
        opt = RMSprop(learning_rate=lr)

    # compile model
    model.compile(loss=model_loss, optimizer=opt, metrics=model_metrics)

    return model


def model_fit(model, training_data):
    history = []
    for i in range(epochs):
        pass
    return history


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

    # load data
    dataset_train = load_data("/train/TFRecord")

    # build model
    model = build_model()
    print(model.summary())
    history = model_fit(model, dataset_train)
    model.save("catsVdogs_model2.h5")
    plot_curves(history)
