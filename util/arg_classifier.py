# All functions for the argument classifier

from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB 
import csv
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd

def get_features_gold_labels(data_type):
    """ Returns a list of common argument dependency paths
    Args:
        data_type: a string of 'dev' / 'test' representing the data type
    
    Returns:
        features: a list of dictionaries of the features
        gold_labels: a list gold labels
    """
    features = []
    gold_labels=[]
    inputfile = 'output/'+data_type+'features.conllu'
    with open(inputfile, encoding='utf-8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='"')
        for row in file:
            if row == []:
                continue
            
            elif row[0].startswith('#') or row[0].startswith('"'):
                continue
            else:
                if row[-1] != '_' and row[-1] != 'V':
                    # set variables for all features
                    token_features = {'token':row[0], 'pos':row[1], 
                                    'prev_token':row[2], 'prev_pos':row[3], 
                                    'next_token': row[4], 'next_pos': row[5],
                                    'head':row[6], 'head_init':row[7], 'predicate': row[8],
                                    'dependency_path':row[9]}
                    gold_label = row[-1]

                    features.append(token_features)
                    gold_labels.append(gold_label)
    
    return features, gold_labels

def create_classifier(training_features, training_labels):
    """ Returns a list of common argument dependency paths
    Args:
        training_features: a list of dictionaries of the features
        training_labels: a list of labels for the training data
    
    Returns:
        model: model fitted with the training features
        vec: a transformer for list of dictionaries to vectors
    """
  
    modeltype = LogisticRegression(max_iter = 400)
        
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(training_features)
    model = modeltype.fit(features_vectorized, training_labels)
    return model, vec


def classify_data(model, inputdata, vec = None):
    """ Use a trained classifier to classify inputdata and write the predictions to a file. Uses extract_features
    Args:
        model: the trained classifier
        inputdata: path to file with input data
        vec: the vectorizer used to transform data
    
    Returns:
        pred_list: list of predicted labels
    """

    features = inputdata
    features = vec.transform(features)
    predictions = model.predict(features)
  
    predlist = []
    for item in predictions:
        predlist.append(item)
    return predlist

