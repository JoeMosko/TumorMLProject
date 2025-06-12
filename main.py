import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.utils import load_img # type: ignore
from tensorflow.keras.utils import img_to_array # type: ignore
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gdown
import shutil
#Source of images
#https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data 


#Need to split the data into...
#70% training
#15% test
#15% validation
#Each folder having 4 subfolders for each type of brain scan

@st.cache_resource
def downloadModel():
    fileId = "1-wt7sEMGElGZ-iLzJKs0Tbtn6gs8uSqm"  # Replace with your actual ID
    url = f"https://drive.google.com/uc?id={fileId}"
    output = "model.keras"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return tf.keras.models.load_model(output)
#model = downloadModel()
@st.cache_resource
def downloadliteModel():
    fileId = "1DiDhdZEM1nkpyWwJUodUqnR0dgCaeHZ0"  # Replace with your actual ID
    url = f"https://drive.google.com/uc?id={fileId}"
    output = "litemodel.keras"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return tf.keras.models.load_model(output)
model = downloadliteModel()
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
classNames = ['glioma', 'healthy', 'meningioma', 'pituitary', 'no scan']


#The shape of each scan (512x512 pixels, rgb 3 colors)
#print(cv2.imread('TrainingData/glioma/0000.jpg').shape)

#Have colors between 0-1

#Create dataset that can be fed to NN

#Indexes for result intepretation
#glioma: 0, healthy: 1, meningioma: 2, pituitary: 3






def predictScan(path, model, classNames):

    #Load the image in the specifified parameters we need (512x512)
    img = load_img(path, target_size=(512,512))

    
    #Convert the image to an arr for the model 
    imgAsArr = img_to_array(img) / 255 #want array between 0 and 1
    
    imgAsArr = np.expand_dims(imgAsArr, axis=0)
    
    #Create an array of final weights predicted by the model 
    predictionProbabilities = model.predict(imgAsArr)
    #Get index of the highest probability 
    predictionIndex = np.argmax(predictionProbabilities)

    #Get the index of the correct prediction of all types of tumors possible in the model 
    predictedClass = classNames[predictionIndex]
    #Take the max of the prediction array (will be confidence %)
    confidence = np.max(predictionProbabilities)
    #Return result to user
    return(f"The model predicts with {confidence * 100:.2f}% confidence that the brain scan shows a {predictedClass} brain.")

def predictUploadedScan(uploadedImg, model, classNames):

    #Load the image in the specifified parameters we need (512x512)
    img = uploadedImg.resize((256, 256))

    
    #Convert the image to an arr for the model 
    imgAsArr = img_to_array(img) / 255 #want array between 0 and 1
    
    imgAsArr = np.expand_dims(imgAsArr, axis=0)
    
    #Create an array of final weights predicted by the model 
    predictionProbabilities = model.predict(imgAsArr)
    #Get index of the highest probability 
    predictionIndex = np.argmax(predictionProbabilities)

    #Get the index of the correct prediction of all types of tumors possible in the model 
    predictedClass = classNames[predictionIndex]
    #Take the max of the prediction array (will be confidence % when multiplied by 100)
    confidence = np.max(predictionProbabilities)

    if (predictedClass == classNames[4]):
        return "The model does not believe this to be an MRI scan. Please try again.", predictionProbabilities
    #Return result to user
    if predictedClass == classNames[1]:
        return (f"The model predicts with {confidence * 100:.2f}% confidence that the brain scan shows no tumor."), predictionProbabilities
    else:
        return(f"The model predicts with {confidence * 100:.2f}% confidence that the brain scan shows a {predictedClass} brain tumor."), predictionProbabilities


def main():
   
    st.markdown('<h1 style="color:#022645; font-size: 40px; text-align: center;"> Tumor Detection by Joey Mosko</h1>', unsafe_allow_html=True)
    #st.markdown('<h1 style="color:#022645;"> Tumor Detection by Joe Mosko</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#022645; font-size: 30px; text-align: center;"> This machine learning model detects glioma, pituitary, meningioma, and pituitary tumors, or if the brain scan shows no tumor by using thousands of sample images to train the model on.</p>', unsafe_allow_html=True)

    
    uploadedFile = st.file_uploader("Start by uploading your image here (jpg or png)", type=["jpg", "jpeg", "png"])
    if uploadedFile is not None:
        img = Image.open(uploadedFile)
        with st.spinner("Analyzing brain scan..."):
            result, probabilities  = predictUploadedScan(img, model, classNames)
        st.write(result)
        if st.button("View graph of probalilities"):
            fig, ax = plt.subplots()
            ax.bar(classNames, np.squeeze(probabilities * 100))
            ax.set_ylabel("Probability %")
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)
        if st.button("View uploaded image"):
            st.image(img.resize((512, 512)))
main()
 

