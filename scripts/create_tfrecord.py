import os
import pathlib
import random

import tensorflow as tf
from PIL import Image

folders = {}
folders["train"] = "data/chest_xray/train"
folders["test"] = "data/chest_xray/test"
folders["validation"] = "data/chest_xray/val"

IMG_SIZE = [224, 224]

validation_ratio = 0.0
test_ratio = 0.0

train_folder = pathlib.Path(folders["train"])
categories = [x.parts[-1] for x in train_folder.iterdir()]

tfrecord_limit = 3557

records_path = "data/"
for name, path in folders.items():
    os.mkdir(records_path + name + "_records/")


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    if iteration == total:
        print()


for name, path in folders.items():
    examples = []
    valid_examples = []
    record_count = 1
    combined_ratio = validation_ratio + test_ratio
    image_folder = pathlib.Path(path)
    for i in categories:
        file_list = list(image_folder.glob(i + "/*.jpeg"))
        n = len(file_list)
        count = 0
        label = categories.index(i)
        printProgressBar(
            0, n - 1, prefix=name + " folder: " + i, suffix="Complete", length=50
        )
        for j in range(0, n):
            img = Image.open(file_list[j], "r")
            img = img.resize((IMG_SIZE[0], IMG_SIZE[1]))
            img.save("test.jpg")
            with open("test.jpg", "rb") as f:
                image_string = f.read()

            # Create a Features message using tf.train.Example.
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image_string])
                        ),
                        "label": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label])
                        ),
                    }
                )
            )

            # Write TF Examples into splits, based on validation and test ratio
            r = random.random()
            if (
                count < (int(n * combined_ratio))
                and r < combined_ratio
                and name == "train"
            ):
                valid_examples.append(example)
            else:
                examples.append(example)
            count += 1

            printProgressBar(
                j, n - 1, prefix=name + " folder: " + i, suffix="Complete", length=50
            )

            # Create multiple TFRecords with ~100MB file size for improved performance
            if len(examples) == tfrecord_limit:
                with tf.io.TFRecordWriter(
                    records_path
                    + name
                    + "_records/record"
                    + str(record_count)
                    + ".tfrecord"
                ) as writer:
                    for e in examples:
                        writer.write(e.SerializeToString())
                    record_count += 1
                    examples.clear()

    with tf.io.TFRecordWriter(
        records_path + name + "_records/record" + str(record_count) + ".tfrecord"
    ) as writer:
        for i in examples:
            writer.write(i.SerializeToString())
        record_count += 1
        examples.clear()

    if len(valid_examples) > 0:
        if validation_ratio > 0.0:
            valid_writer = tf.io.TFRecordWriter(
                records_path + "validation_record.tfrecord"
            )
        if test_ratio > 0.0:
            test_writer = tf.io.TFRecordWriter(records_path + "test_record.tfrecord")
        for i in valid_examples:
            if random.random() < (validation_ratio / combined_ratio):
                valid_writer.write(i.SerializeToString())
            else:
                test_writer.write(i.SerializeToString())
        valid_examples.clear()
