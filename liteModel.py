import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import os
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
train = ImageDataGenerator(
    rescale=1/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

#Create dataset that can be fed to NN
trainData = train.flow_from_directory(
    'TrainingData',
    target_size=(256,256),
    class_mode='categorical',
    batch_size=32,
    shuffle=True
)
validationData = validation.flow_from_directory(
    'ValidationData',
    target_size=(256,256),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)
TestData = test.flow_from_directory(
    'TestingData',
    target_size=(256,256),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

print('Class indices:', trainData.class_indices)

#create model 
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256,256,3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])

#compile model 
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_litemodel.keras', save_best_only=True, monitor='val_loss')
]

#fit the model with the training data
modelFit = model.fit(
    trainData,
    epochs=15,
    validation_data=validationData,
    callbacks=callbacks
)

#Evaluate loss and accuracy 
TestData.reset()
testLoss, testAccuracy = model.evaluate(TestData)

print(f"Test Accuracy: {testAccuracy * 100:.2f}%")
print(f"Test Loss: {testLoss:.4f}")

model.save("litemodel.keras")
