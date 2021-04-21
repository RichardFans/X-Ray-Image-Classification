# Imports
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


def get_predictions(input_image):
    tflite_interpreter = tf.lite.Interpreter(model_path="demo/quantized_model.tflite")
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    tflite_interpreter.allocate_tensors()
    tflite_interpreter.set_tensor(input_details[0]["index"], input_image)
    tflite_interpreter.invoke()
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    return tflite_model_prediction


## Page Title
st.set_page_config(page_title="X Ray Image Classification", page_icon="🩺")
st.title("🩺 X Ray Image Classification")
st.markdown("---")

## Sidebar
st.sidebar.header("🩺 X Ray Image Classification")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "If you liked this project and would like to read the code and see some of my other work, don't forget to ⭐the [repository](https://github.com/SauravMaheshkar/X-Ray-Image-Classification) and follow [me](https://github.com/SauravMaheshkar)."
)


st.header("Interactive Demo")
st.info(
    "NOTE: This demo uses a quantized EfficientNetB0 model for running inference. The code used for training the model can be found in the [github repository](https://github.com/SauravMaheshkar/X-Ray-Image-Classification)"
)
## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg", "png"])
if uploaded_file is not None:
    image_pred = np.asarray(bytearray(uploaded_file.read()))
    image_pred = Image.fromarray(image_pred).convert("RGB")
    open_cv_image = np.array(image_pred)
    image_pred = cv2.resize(open_cv_image, (512, 512))
    img = np.expand_dims(image_pred, axis=0).astype(np.float32)

if st.button("Get Predictions"):
    suggestion = get_predictions(input_image=img)
    st.write(suggestion)

st.markdown("---")
st.markdown(
    "If you liked this project and would like to read the code and see some of my other work, don't forget to ⭐the [repository](https://github.com/SauravMaheshkar/X-Ray-Image-Classification) and follow [me](https://github.com/SauravMaheshkar)."
)
