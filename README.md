<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<p align="center">
  <a href="https://github.com/github_username/repo">
    <img src="demo/cover.svg" alt="Logo" width="40%">
  </a>

  <h3 align="center">Image Processing API</h3>

  <p align="center">
    a python based image processing api
  </p>
</p>


<!-- ABOUT THE PROJECT -->

## About The Project

![](/demo/landmark.png)

A python based image processing api.


### Built With

* [Flask](https://github.com/pallets/flask)
* [Tensorflow](https://github.com/tensorflow/tensorflow)
* [Keras](https://github.com/keras-team/keras)
* [OpenCV](https://github.com/opencv/opencv)
* [numpy](https://github.com/numpy/numpy)
* [matplotlib](https://github.com/matplotlib/matplotlib)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

#### Running locally
* Install the [dependencies](requirements.txt)
* Run ``` python server.py ``` 

## Usage
GET to ```localhost:8080/api/marks?image=https://raw.githubusercontent.com/AlissonSteffens/image-processing-api/master/demo/lenna.jpg```

### Endpoints

#### Knowledge API
* **/api/faces** - face rects of image
* **/api/marks** - 68 main landmarks on faces
* **/api/direction** - information about here is the person looking 
* **/api/emotion** - predicts the person emotion 

#### Other
* **/marks** - 68 main landmarks on faces
* **/face** - return the most important face as a image

Alternatively you can use:
  "marker_size=x" to set marker size.
  "image_size=x" to set image size.
