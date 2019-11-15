import tensorflow as tf
from matplotlib.image import imread
import glob
import os


# helper functions
def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(image_paths, labels, out_path):
    # image_paths   list of paths for the images
    # labels        list of class labels for the images
    # out_path      path to the TFRecords output file

    # Number of images. Used when printing the progress
    num_images = len(image_paths)

    # Open a TFRecordWriter for the output file
    with tf.io.TFRecordWriter(out_path) as writer:

        # Iterate over all the images and labels
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            # Print progress.
            if i % 100 == 0:
                print(f"\tConverting images {i}")

            # Load image
            img = imread(path)
            # Convert the image to raw bytes
            img_bytes = img.tostring()

            # Create a dict with the data wanted to save in the TFRecords file
            data = {
                'image': wrap_bytes(img_bytes),
                'label': wrap_int64(label)
            }

            # Wrap the data as TensorFlow Features
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example
            example = tf.train.Example(features=feature)

            # Serialize the data
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file
            writer.write(serialized)


def makeTFRecords(dir):

    files = os.listdir(f"./{dir}/{dir}_tfrecord")
    if files:
        for file in files:
            print(f"Deleting files in ./{dir}/{dir}_tfrecord/{file}")
            os.remove(os.path.join(f"./{dir}/{dir}_tfrecord", file))

    print(f"Making {dir} set of TFRecords:")
    files = glob.glob(f"./{dir}/*/*")
    labels = [0 if "cat" in x else 1 for x in files]
    convert(files, labels, f"./{dir}/{dir}_tfrecord/{dir}.tfrecord")


if __name__ == "__main__":
    makeTFRecords("train")
    makeTFRecords("test")
    makeTFRecords("val")
