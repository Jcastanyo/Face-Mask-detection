# Face-Mask-detection
Face mask detection using OpenCV face detection and VGG19 for a final mask-nomask classification.

I did this project to put into practice theorical knowledge that I learned reading François Chollet's book about Deep Learning with Keras and Python.
The main goal is to train a Convolutional Neural Network, VGG19 in this case, to predict if a person is wearing a mask or not. This neural network does not predict bounding boxes, so I had to use a pre-trained neural network from OpenCV to detect faces first.

I uploaded train, test and real time test code. The comments are in spanish, I am sorry. Nonetheless, the code is very easy if you already know something about Keras, Python and OpenCV. 

The dataset I used to train the CNN is called FaceMaskDetection Dataset from Kaggle (https://www.kaggle.com/omkargurav/face-mask-dataset). I don't know the url to download the OpenCV's face detector, but there are like three or four different face detectors both in OpenCV and Dlib library, maybe Dlib's face detector is a better option.

In this directory, you can also find a kind of summary of the project, it is also written in spanish, and two images to see how the training went. 

A video with the results is available in my youtube channel: https://youtu.be/C6mNvurEvYA. The detection is a little bit slow because my hardware is not really fast. 
