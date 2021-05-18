# 🩺 X-Ray-Image-Classification

Comparative Analysis of Convolutional Neural Networks for X Ray 🩺 Image Analysis. Check out the [Deepnote Kernel](https://deepnote.com/@saurav-maheshkar/X-Ray-Image-Classification-M7ID1ABnTqysUd7USuIaUw) used for modeling and experimentation.

![](https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/xray-app.gif?raw=true)

[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)


A Complete End to End Deep Learning Project, going over Modelling, Ablation Studies, Post-Training Quantization and Deployment using Streamlit and Tensorflow Serving.

| Model Type                           | Total Parameters | Trainable Parameters | Non-Trainable Parameters |
|--------------------------------------|------------------|----------------------|--------------------------|
| EfficientNetB0 + Classification Head | 4,059,828        | 10,257               | 4,049,571                |

The Data used in this project is available at the [Activeloop platform](https://app.activeloop.ai/) in 3 sub-splits. Each subset can be found at:

* [Training Subset](https://app.activeloop.ai/datasets/explore?tag=sauravmaheshkar%2Fchest_xray_pneumonia_train)
* [Validation Subset](https://app.activeloop.ai/datasets/explore?tag=sauravmaheshkar%2Fchest_xray_pneumonia_val)
* [Test Subset](https://app.activeloop.ai/datasets/explore?tag=sauravmaheshkar%2Fchest_xray_pneumonia_test)

# Developer Toolkit

![](https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/Toolkit.png?raw=true)

# Metrics

[![Explore-in W&B](https://img.shields.io/badge/Explore--in-W%26B-%23FFBE00)](https://wandb.ai/sauravmaheshkar/xray-image-classification)

Weights and Biases client was used for model monitoring, for more details please visit the project page.

![](https://raw.githubusercontent.com/SauravMaheshkar/X-Ray-Image-Classification/5a29b9fd7cf1f2697866aca38c875d43ee6ec5b0/assets/Validation%20AUC.svg)

![](https://raw.githubusercontent.com/SauravMaheshkar/X-Ray-Image-Classification/5a29b9fd7cf1f2697866aca38c875d43ee6ec5b0/assets/Validation%20Loss.svg)

# Exploratory Data Analysis

<details> 
<summary>Label Distribution</summary>
 <center><img src = 'https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/eda/label_distribution.png?raw=true'></center>
</details>

<details> 
<summary>Ben Graham's Method</summary>
 <center><img src = 'https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/eda/bengraham.png?raw=true'></center>
</details>

<details> 
<summary>Background Subtraction</summary>
 <center><img src = 'https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/eda/bg_sub.png?raw=true'></center>
</details>

<details> 
<summary>Fourier Method for Pixel Distribution</summary>
 <center><img src = 'https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/eda/fourier.png?raw=true'></center>
</details>

<details> 
<summary>Canny Edge Detection</summary>
 <center><img src = 'https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/eda/canny.png?raw=true'></center>
</details>

# GradCAM

![](https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/gradcam.png?raw=true)

The Colab Notebook used for obtaining this image can be found in the `notebooks/` folder.

# Docker Instructions

```
docker pull docker.pkg.github.com/sauravmaheshkar/x-ray-image-classification/xray-streamlit:v0.0.1
docker run -p 8501:8501 xray-streamlit:latest
```

# Contribute

If you want to contribute to the project kindly mail me at `sauravvmaheshkar@gmail.com`.

### Step 1
 - **Option 1**
   🍴 Fork it!  
 - **Option 2**
    👯‍♂️ Clone this repo to your local machine using `https://github.com/SauravMaheshkar/X-Ray-Image-Classification.git`
### Step 2

- **HACK AWAY!** 🔨🔨🔨

### Step 3

- 🔃 Create a new pull request using `https://github.com/SauravMaheshkar/X-Ray-Image-Classification/compare/`


# License

[![License](http://img.shields.io/:license-mit-blue.svg)](http://doge.mit-license.org)

The data for this project was taken from kaggle datasets. You can find the Chest X-Ray Images (Pneumonia) 
Dataset [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

- Copyright 2020 @[Saurav Maheshkar](https://sauravmaheshkar.github.io/)
- [MIT License](https://opensource.org/licenses/MIT)


# Credits

The inspiration for this readme file came from
- [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46#license)
- [Navendu Pottekkat](https://github.com/navendu-pottekkat/awesome-readme/blob/master/README-template.md)
