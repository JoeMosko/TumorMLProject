import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2 
import shutil
# https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data 


'''
Need to split the data into...
70% training
15% test
15% validation
Each folder having 4 subfolders for each type of brain scan
'''


def moveHealthyFilesToTraining():

    #For healthy...
    SourceDir = "./images/healthy/"
    TrainingDir = "./TrainingData/healthy/"

    #Error handling (making sure both folders exist)
    if not os.path.exists(SourceDir):
        print(SourceDir, "does not exist")
        return 
    if not os.path.exists(TrainingDir):
        print(TrainingDir, "does not exist")
        return 

    Files = os.listdir(SourceDir)
    #Calculate 75 percent of the files to be moved
    numToMove = int(len(Files) * .75)
    #Parse 75 percent of the files to a new array
    selectedFiles = Files[:numToMove]

    #loop through all files in selectedFiles
    for file in selectedFiles:
        Path = os.path.join(SourceDir, file)
        Destination = os.path.join(TrainingDir, file)
        #Move from old path to new dir
        shutil.move(Path, Destination)
    print(f"Completed transfer of {len(selectedFiles)} files of healthy brain scans to training data")

def moveGliomaFilesToTraining():
    #For healthy...
    SourceDir = "./images/glioma/"
    TrainingDir = "./TrainingData/glioma/"

    #Error handling (making sure both folders exist)
    if not os.path.exists(SourceDir):
        print(SourceDir, "does not exist")
        return 
    if not os.path.exists(TrainingDir):
        print(TrainingDir, "does not exist")
        return 

    Files = os.listdir(SourceDir)
    #Calculate 75 percent of the files to be moved
    numToMove = int(len(Files) * .75)
    #Parse 75 percent of the files to a new array
    selectedFiles = Files[:numToMove]

    #loop through all files in selectedFiles
    for file in selectedFiles:
        Path = os.path.join(SourceDir, file)
        Destination = os.path.join(TrainingDir, file)
        #Move from old path to new dir
        shutil.move(Path, Destination)
    print(f"Completed transfer of {len(selectedFiles)} files of glioma brain scans to training data")

def moveMeningiomaFilesToTraining():
    #For healthy...
    SourceDir = "./images/meningioma/"
    TrainingDir = "./TrainingData/meningioma/"

    #Error handling (making sure both folders exist)
    if not os.path.exists(SourceDir):
        print(SourceDir, "does not exist")
        return 
    if not os.path.exists(TrainingDir):
        print(TrainingDir, "does not exist")
        return 

    Files = os.listdir(SourceDir)
    #Calculate 75 percent of the files to be moved
    numToMove = int(len(Files) * .75)
    #Parse 75 percent of the files to a new array
    selectedFiles = Files[:numToMove]

    #loop through all files in selectedFiles
    for file in selectedFiles:
        Path = os.path.join(SourceDir, file)
        Destination = os.path.join(TrainingDir, file)
        #Move from old path to new dir
        shutil.move(Path, Destination)
    print(f"Completed transfer of {len(selectedFiles)} files of meningioma brain scans to training data")

def movePituitaryFilesToTraining():
    #For Pituitary...
    SourceDir = "./images/pituitary/"
    TrainingDir = "./TrainingData/pituitary/"

    #Error handling (making sure both folders exist)
    if not os.path.exists(SourceDir):
        print(SourceDir, "does not exist")
        return 
    if not os.path.exists(TrainingDir):
        print(TrainingDir, "does not exist")
        return 

    Files = os.listdir(SourceDir)
    #Calculate 75 percent of the files to be moved
    numToMove = int(len(Files) * .75)
    #Parse 75 percent of the files to a new array
    selectedFiles = Files[:numToMove]

    #loop through all files in selectedFiles
    for file in selectedFiles:
        Path = os.path.join(SourceDir, file)
        Destination = os.path.join(TrainingDir, file)
        #Move from old path to new dir
        shutil.move(Path, Destination)
    print(f"Completed transfer of {len(selectedFiles)} files of pituitary brain scans to training data")

#moveGliomaFilesToTraining()

#moveHealthyFilesToTraining()

#moveMeningiomaFilesToTraining()

#movePituitaryFilesToTraining()




#Needed to interpret output of the model 
classNames = ['glioma', 'healthy', 'meningioma', 'not MRI scan', 'pituitary']


#The shape of each scan (512x512 pixels, rgb 3 colors)
# print(cv2.imread('TrainingData/glioma/0000.jpg').shape)

#Have colors between 0-1
train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)
test = ImageDataGenerator(rescale = 1/255)

#Create dataset that can be fed to NN

trainData = train.flow_from_directory('TrainingData', 
                                      #keep size the same 
                                      target_size = (256,256),
                                      #Need categorical class because our output is more than 2 classes 
                                      class_mode = 'categorical',
                                      batch_size = 32)
validationData = validation.flow_from_directory('ValidationData',
                                                #keep size the same 
                                                target_size = (256,256),
                                                #Need categorical class because our output is more than 2 classes 
                                                class_mode = 'categorical',
                                                batch_size = 32)
TestData = validation.flow_from_directory('TestingData',
                                                #keep size the same 
                                                target_size = (256,256),
                                                #Need categorical class because our output is more than 2 classes 
                                                class_mode = 'categorical',
                                                batch_size = 32)
#Indexes for result intepretation
#glioma: 0, healthy: 1, meningioma: 2, pituitary: 3, no scan: 4

#create model 

model = tf.keras.Sequential([
    #first layer (transforms format of image from a 2d array to a 1d array)
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256,256,3)),
    
    tf.keras.layers.MaxPool2D(2,2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(64, activation='relu'), 
    #prevent overfitting - where the model becomes innacurate based on too much data 
    tf.keras.layers.Dropout(0.5),
    #4 and softmax activation because we have 4 outcomes 
    tf.keras.layers.Dense(5, activation='softmax')
])

#compile model 
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#fit the model with the training data
modelFit = model.fit(trainData, epochs=7, validation_data = validationData)

probablityModel  = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
#Evaluate loss and accuracy 
testLoss, testAccuracy = model.evaluate(TestData)

print(f"Test Accuracy: {testAccuracy * 100:.2f}%")
print(f"Test Loss: {testLoss:.4f}")


model.save("litemodel.keras")