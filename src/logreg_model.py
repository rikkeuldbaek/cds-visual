# import packages

# path tools
import os
import cv2

# data loader
import numpy as np
from tensorflow.keras.datasets import cifar10

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# classificatio models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

######### LOGISTIC REGRESSION MODEL ############

def logreg_model_function:
    
    # Read in the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    labels = ["airplane", 
          "automobile", 
          "bird", 
          "cat", 
          "deer", 
          "dog", 
          "frog", 
          "horse", 
          "ship", 
          "truck"] #note this is alfabetically 

    # Convert to greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # Scaling
    X_train_scaled = (X_train_grey)/255.0 
    X_test_scaled = (X_test_grey)/255.0 

    # Reshape training data
    nsamples, nx, ny = X_train_scaled.shape #extracting each dimension (samples, 32, 32)
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    # Reshape test data 
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))


