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
training_filenames = "./TFRecords/train_tfrecord/train.tfrecord"
validation_filenames = "./TFRecords/val_tfrecord/val.tfrecord"
test_dir = "./TFRecords/test_tfrecord/test.tfrecord"

# hypter parameters
batch_size = 32
epochs = 100
steps_per_epoch = 32
validation_steps = 32

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


def read_tfrecord(example):
    features = {
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
        # tf.string = bytestring (not text string)
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }

    # decode the TFRecord
    parsed_example = tf.io.parse_single_example(example, features)

    # Get the label & dimesnsions of the image
    label = tf.cast(parsed_example['label'], tf.int64)
    width = tf.cast(parsed_example['width'], tf.int64)
    height = tf.cast(parsed_example['height'], tf.int64)
    channels = tf.cast(parsed_example['channels'], tf.int64)

    # Get the image as raw bytes
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.io.decode_raw(image_raw, tf.uint8)
    # scale back to original image
    image_shape = tf.stack([height, width, channels])
    image = tf.reshape(image, image_shape)
    # resize required
    image = tf.image.resize(image, [input_size[0], input_size[1]])

    return image, label


def get_batched_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord)

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    return dataset


def get_training_dataset():
    return get_batched_dataset(training_filenames)


def get_validation_dataset():
    return get_batched_dataset(validation_filenames)


if __name__ == "__main__":
    model = build_model()
    print(model.summary())
    history = model.fit(get_training_dataset(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=get_validation_dataset(),
                        validation_steps=validation_steps)
    model.save("catsVdogs_model1.h5")
    plot_curves(history)
