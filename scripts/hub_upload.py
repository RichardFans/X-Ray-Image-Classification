"""
* Example of generating hub.Dataset for Image Classification
using @hub.transform

* Link to the original dataset:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

* Ported from Hub/examples:
https://github.com/activeloopai/Hub/blob/master/examples/generate_ds_transform.py
"""
import glob
import os

import hub
import numpy as np
import PIL.Image
from hub.schema import ClassLabel, Image

# skipcq: PYL-W0105
"""Create a new dataset

|------------|------|
| Split      | #    |
|------------|------|
| Train      | 5216 |
| Test       | 624  |
| Validation | 16   |
|------------|------|"""


schema = {
    "image": Image(shape=(None, None, None), max_shape=(3000, 3000, 3), dtype="uint8"),
    "label": ClassLabel(num_classes=2),
}
tag = "sauravmaheshkar/chest_xray_pneumonia_test"
len_ds = 624
ds = hub.Dataset(tag, mode="w+", shape=(len_ds,), schema=schema)


# Transform function
@hub.transform(schema=schema, scheduler="threaded", workers=8)
def fill_ds(filename):
    if os.path.basename(os.path.dirname(filename)) == "NORMAL":
        label = 0
    else:
        label = 1
    image = np.array(PIL.Image.open(filename))
    if len(image.shape) == 2:
        image = np.expand_dims(image, -1)
    return {
        "image": image,
        "label": label,
    }


# Fill the dataset and store it
file_list = glob.glob("data/chest_xray/test/*/*.jpeg")
ds = fill_ds(file_list)
ds = ds.store(tag)
ds.flush()
