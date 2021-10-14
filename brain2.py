from tkinter import filedialog
from imageai.Prediction import ImagePrediction
import os
import tkinter as tk
from tkinter import *
execution_path=os.getcwd()

"""
Browse for an image file
display top 3 predicted image classifications
compares mobilenet, resnet, inception (google) 
and densenet (facebook) models"""


def predict(prediction, image_file):
    pred = {}
    prediction.loadModel()
    predictions, probabilities = prediction.classifyImage(image_file, result_count=3)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        pred.update({eachPrediction:eachProbability})
    return pred


def open_file(path):
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir=path, title="Select file",
                filetypes=(("jpeg files", "*.jpg"),("png files", "*.png")))
    return root.filename


image_file = open_file(execution_path)

#Prediction
results = []
prediction = ImagePrediction()

prediction.setModelTypeAsMobileNetV2() #requires pytorch from conda
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2.h5"))
results.append(predict(prediction, image_file))

prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50_imagenet_tf.2.0.h5"))
results.append(predict(prediction, image_file))

prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
results.append(predict(prediction, image_file))

prediction.setModelTypeAsDenseNet121()
prediction.setModelPath(os.path.join(execution_path, "DenseNet-BC-121-32.h5"))
results.append(predict(prediction, image_file))

print('\nImage prediction results:')
print('\nmobilenet_v2.h5', results[0])
print('\nresnet50_imagenet_tf.2.0.h5', results[1])
print('\ninception_v3_weights_tf_dim_ordering_tf_kernels.h5', results[2])
print('\nDenseNet-BC-121-32.h5', results[3])