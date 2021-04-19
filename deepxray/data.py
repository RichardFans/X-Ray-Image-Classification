from functools import partial

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class TFRecordsDataset:
    def __init__(self, image_size: list, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size

    def _decode_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [*self.image_size, 3])
        return image

    def read_tfrecord(self, example, labeled):
        tfrecord_format = (
            {
                "image": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64),
            }
            if labeled
            else {
                "image": tf.io.FixedLenFeature([], tf.string),
            }
        )
        example = tf.io.parse_single_example(example, tfrecord_format)
        image = self._decode_image(example["image"])
        if labeled:
            label = tf.cast(example["label"], tf.int64)
            return image, label
        return image

    def load_dataset(self, filenames, labeled=True):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.map(
            partial(self.read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
        )
        return dataset

    def get_dataset(self, filenames, labeled=True):
        dataset = self.load_dataset(filenames, labeled=labeled)
        dataset = dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        return dataset
