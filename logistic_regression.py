# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:27:27 2015

@author: Haley
"""
from sklearn.linear_model import LogisticRegression
import numpy

# YOU ARE GIVEN THIS HELPER FUNCTION
def pick_features(data_matrix, feature_list):
    """
    data_matrix: 2D-array data matrix, where len(data_matrix) is the number of 
                 examples, and and len(data_matrix[0]) is the number of features
    feature_list: a list of feature indicies to choose

    Takes in a data matrix and a list of feature indicies, and returns 
    a new data matrix that includes only the features in feature_list
    """
    return data_matrix[:, feature_list]
  
# IMPLEMENT THIS FUNCTION
def compare_models(train_data, train_labels, test_data_list, test_labels_list, features_a, features_b):
    """
    train_data: 2D-array of training data that has a matrix shape
                where len(train_data) is number of examples and 
                len(train_data[0]) is number of features
    train_labels: 1D-array of labels for the training set where 
                  len(train_labels) is the number of examples
    test_data_list: A list of 2D-arrays of test data where each array 
                    has a matrix shape where len(test_data_list[i]) is number of 
                    examples and len(train_data_list[i][0]) is number of features
    test_labels_list: A list of 1D-arrays of test data where each array's shape 
                      is len(test_labels_list[i]) is the number of examples
    features_a: list of feature indices to include for model A
    features_b: list of feature indices to include for model B
  
    Trains two logistic regression models on the train_data, once with 
    features_a, once with features_b. Tests both models on the testing data.

    Returns: a tuple of 
             (1) the average accuracy of the model trained with features_a, 
             (2) the average accuracy of the model trained with features_b
    """
    
    #Model A
    training_data_a = pick_features(train_data, features_a)
    
    model_a = LogisticRegression()
    fit_a = model_a.fit(training_data_a, train_labels)
    accuracies_a = []
    
    for i in xrange(len(test_data_list)):
        labels = []
        truep = 0
        falsep = 0
        truen = 0
        falsen = 0
        probmatrix_a = fit_a.predict_proba(test_data_list[i])
        for j in probmatrix_a[0]:
            if j > 0.5:
                labels.append(1)
            else:
                labels.append(0)
        for k in xrange(len(labels)):
            if labels[k] == test_labels_list[i][k]:
                if labels[k] == 0:
                    truep += 1
                else:
                    truen += 1
            else:
                if labels[k] == 1:
                    falsep += 1
                else:
                    falsen += 1
        num = truep + truen
        dem = truep + truen + falsep + falsen
        accuracy = num / dem
        accuracies_a.append(accuracy)
    
    total = 0
    for a in accuracies_a:
        total += a
    
    avg_accuracy_a = total / len(accuracies_a)

    
    #Model B
    training_data_b = pick_features(train_data, features_b)
    
    model_b = LogisticRegression()
    fit_b = model_b.fit(training_data_b, train_labels)
    accuracies_b = []
    
    for i in xrange(len(test_data_list)):
        labels = []
        truep = 0
        falsep = 0
        truen = 0
        falsen = 0
        probmatrix_b = fit_b.predict_proba(test_data_list[i])
        for j in probmatrix_b[0]:
            if j > 0.5:
                labels.append(1)
            else:
                labels.append(0)
        for k in xrange(len(labels)):
            if labels[k] == test_labels_list[i][k]:
                if labels[k] == 0:
                    truep += 1
                else:
                    truen += 1
            else:
                if labels[k] == 1:
                    falsep += 1
                else:
                    falsen += 1
        num = truep + truen
        dem = truep + truen + falsep + falsen
        accuracy = num / dem
        accuracies_b.append(accuracy)
    
    total = 0
    for b in accuracies_b:
        total += b
    
    avg_accuracy_b = total / len(accuracies_b)
    
    return (avg_accuracy_a, avg_accuracy_b)
    
    
data = numpy.array([[0,0,0], [0,0,0], [1,1,1], [1,0,1], [0,0,0], [1,1,0]])
labels = numpy.array([0,0,1,1,0,1])
train_data = data[0:3, :]
train_labels = labels[0:3]
    
test_data_list = []
test_labels_list = []

for i in range(3, 6):
    test_data_list.append(data[i:i+1, :])
    test_labels_list.append(labels[i:i+1])
print test_data_list[1][1]
    
features_a = [0,1,2]
features_b = [0,1,2]

#print compare_models(train_data, train_labels, test_data_list, test_labels_list, features_a, features_b)
    
#print pick_features(data, features_