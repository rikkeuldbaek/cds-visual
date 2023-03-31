# import packages

# path tools
import os
import cv2

# data loader
import numpy as np

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# classificatio models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Save model
from joblib import dump, load

# Scripting
import argparse


##### Arguments #####

def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()
    #add arguments
    parser.add_argument("--penalty", type= str, default="none")
    parser.add_argument("--tol", type= float, default=0.1)
    parser.add_argument("--verbose", type= bool, default="True")
    parser.add_argument("--solver", type= str, default="saga")
    parser.add_argument("--multi_class", type= str, default="multinomial")
    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value
    return args #returning arguments


######### LOGISTIC REGRESSION MODEL ############

def logreg_model_function(penalty, tol, verbose, solver, multi_class):

    clf = LogisticRegression(penalty="none", 
                        tol=0.1,
                        verbose=True,
                        solver="saga",
                        multi_class="multinomial").fit(X_train_dataset, y_train)

    y_pred = clf.predict(X_test_dataset)

    report = classification_report(y_test, 
                               y_pred, 
                               target_names=labels) #cool classification report function
    
    print(report)


    # Save the classification report in the folder "out"
    # Define out path
    outpath_metrics_report = os.path.join(os.getcwd(), "out", "LR_metrics_report.txt")

    # Save the metrics report
    file = open(outpath_metrics_report, "w")
    file.write(report)
    file.close()

    # Save the trained model to the folder called "models"
    # Define out path
    outpath_classifier = os.path.join(os.getcwd(), "models", "LR_classifier.joblib")

    # Save model
    dump(clf, open(outpath_classifier, 'wb'))

    print( "Saving the logistic regression metrics report in the folder ´out´")
    print( "Saving the logistic regression model in the folder ´models´")



# Define a main function
def main():
    # input parse
    args = input_parse()
    # pass arguments to logistic regression function
    logreg_model_function(args.penalty, args.tol, args.verbose, args.solver, args.multi_class)


