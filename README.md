# image-processing-api
a python based image processing api

![](demo/landmark.png)

## Usage
POST to ```localhost:8080/face-marks``` *send the image as base64 in the request body*

## Running locally
* Install de [dependencies](requirements.txt)
* Run ``` python server.py ``` 
  
## Running locally with Docker
* Build ``` docker build . -t image-processing-api```
* Run ``` docker run -p 8080:8080 image-processing-api ```
