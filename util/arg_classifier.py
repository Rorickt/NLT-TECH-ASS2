# All functions for the argument classifier

from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB 
import csv
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd

def create_classifier(training_features, training_labels, modelname):
    """
    
    
    """
  
    modeltype = LogisticRegression(max_iter = 400)
        
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(training_features)
    model = modeltype.fit(features_vectorized, training_labels)
    return model, vec


def classify_data(model, inputdata, outputfile, vec = None):
    """
    Use a trained classifier to classify inputdata and write the predictions to a file. Uses extract_features
    
    model: the trained classifier
    type model:
    vec: the vectorizer used to transform data
    type vec:
    inputdata: path to file with input data
    type inputdata: string
    outputfile: path to file where output should be stored
    type outputfile: string
    
    return: None
"""
    features = inputdata
    features = vec.transform(features)
    predictions = model.predict(features)
  
    predlist = []
    for item in predictions:
        predlist.append(item)
    return predlist