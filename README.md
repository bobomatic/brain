# brain
Tensorflow based Image recognition tool using machine learning. Compares ability of facebook, google and other image recognition algorithms to recognise a jpg or png image.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Examples of Use](#examples-of-use)
* [Status](#status)
* [Sources](#sources)

## General Info
This project uses the tensorflow machine learning library to identify an image. 4 different models are used so that the user can compare their effectiveness.

## Technologies
This project is created with

XXXXXX

## Setup
To run this project install it locally using git clone.

```
$ cd ../lorem
$ git clone <paste address of repository>
$ pip install -r requirements.txt
```

## Features
* User chooses an image file (tkinter file dialog)
* Brain classifies the image using 4 different prediction models:
  - mobilenet_v2.h5
  - resnet50_imagenet_tf
  - inception_v3 (google)
  - Densenet (facebook)
* For each model the top 3 classifications are displayed alongside their probability

### To do:
Make a windows executable version.

## Examples of Use

Usage: 

python3 brain2.py

Code example:

Command line
`$ python3 brain2.py`

## Status
Complete

## Sources
This project is inspired by Andrei Neagoie Python Zero to Mastery course:

https://www.udemy.com/course/complete-python-developer-zero-to-mastery
