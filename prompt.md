I need this mvp built in an hour. This is a computer application that can control a computer using your camera. 

It will be in python

Make sure your machine learning solution can identify a stream of frames and classify based on a dynamic amount of information. The data will be singular hand pictures and we will need to also allow a user to cursor.

This will be how we take the data and use it to classify.

we will take hand gesture data (captured in mediapipe) and downloaded the data to be machine learning trained for classification. We will need to transform the data into machine code so that it can be easily trained.

capture.py will process our data, overlay mediapipe, and reduce data down to its machine code format
train.py will create a model based on this input data
app.py -> opens a camera on the computer and lets user control computer using their hands
Using pyautogui we will control the computer and map gestures to tasks

Create all three files and make it work the first try after inputting the data