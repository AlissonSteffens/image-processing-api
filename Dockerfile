FROM python:3.7
MAINTAINER AlissonSteffens

RUN pip install Flask
RUN pip install Pillow
RUN pip install numpy
RUN pip install opencv-python
RUN pip install cmake
RUN pip install dlib


ENV APP_HOME /app
COPY . $APP_HOME
WORKDIR $APP_HOME

ENTRYPOINT python server.py
EXPOSE 8080