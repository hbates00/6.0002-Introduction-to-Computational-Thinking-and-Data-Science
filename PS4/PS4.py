# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 00:15:13 2015

@author: belinkov, nrakover
"""
#PS4
#Haley Bates-Tarasewicz
#Collaborators: XhiDae, Same
#Time Spent: 8:30

import random
import pylab
import sklearn
from sklearn.linear_model import LogisticRegression

#### Helper code: you do not need to change anything in this part

# default values to be used unless specified differently
DEFAULT_CORRUPT_FRAC = 0.05
DEFAULT_HOLDOUT_FRAC = 0.2
DEFAULT_THRESHOLD = 0.5
FILENAME = 'tumorInfo.txt'


def corruptData(data, corrupt_frac):
    """
    Returns a copy of the data where roughly corrupt_frac of the
    values have been overwritten with random numbers

    data (list): a list of strings, each string encoding the
            ID, label, and features of a tumor
    corrupt_frac (float): a float between 0 and 1, determining the
            probability that a given feature value will be corrupted

    Returns a corrupted version of the data
    """
    newData = []
    for line in data:  
        fields = line.split(',')
        newLine = fields[0] + ',' + fields[1]
        for i in range(2, len(fields)):
            fields[i] = float(fields[i])
            newLine = newLine + ','
            if random.random() < corrupt_frac:
                fields[i] = round(random.gauss(0, 100.0), 5)
            newLine = newLine + str(fields[i])
        newData.append(newLine)
    return newData
       
def printStats(truePos, falsePos, trueNeg, falseNeg, spaces = ''):
    """
    Pretty-print the true/false negatives/positives
    """
    print spaces + 'Accuracy =', accuracy(truePos, falsePos, trueNeg, falseNeg)
    print spaces + 'Sensitivity =', sensitivity(truePos, falseNeg)
    print spaces + 'Specificity =', specificity(trueNeg, falsePos)
    print spaces + 'Pos. Pred. Val. =', posPredVal(truePos, falsePos)

class Tumor(object):
    """
    Wrapper for the tumor data points
    """
    def __init__(self, idNum, malignant, featureNames, featureVals):
        self.idNum = idNum
        self.label = malignant
        self.featureNames = featureNames
        self.featureVals = featureVals
    def distance(self, other):
        dist = 0.0
        for i in range(len(self.featureVals)):
            dist += abs(self.featureVals[i] - other.featureVals[i])**2
        return dist**0.5
    def getLabel(self):
        return self.label
    def getFeatures(self):
        return self.featureVals
    def getFeatureNames(self):
        return self.featureNames
    def __str__(self):
        return str(self.idNum) + ', ' + str(self.label) + ', ' \
               + str(self.featureVals)

def getTumorData(inData, dontUse = []):
    """
    Parses each data point in inData into an instance of the Tumor class,
    where the features listed in dontUse are omitted

    inData (list): a list of strings, each string encoding the
            ID, label, and features of a tumor
    dontUse (list): a list of strings, each the name of a feature to omit

    Returns a list of Tumor instances built from the data points provided
            and also returns the list of the names of the features used
    """
    means = ['radiusMean', 'textureMean', 'perimeterMean', 'areaMean',\
             'smoothnessMean', 'compactnessMean', 'concavityMean',\
             'concavePtsMean', 'symmetryMean', 'fractalDMean']
    stdErrs = ['radiusSE', 'textureSE', 'perimeterSE', 'areaSE', \
               'smoothnessSE', 'compactnessSE', 'concavitySE', 
               'concavePtsSE', 'symmetrySE', 'fractalDSE']
    worsts = ['radiusWorst','textureWorst', 'perimeterWorst', 'areaWorst',\
              'smoothnessWorst', 'compactnessWorst', 'concavityWorst',\
              'concavePtsWorst','symmetryWorst', 'fractalDWorst']
    possibleFeatures = means + stdErrs + worsts
    data = []
    for line in inData:
        split = line.split(',')
        idNum = int(split[0])
        if split[1] == 'B':
            malignant = 0
        elif split[1] == 'M':
            malignant = 1
        else:
            raise ValueError('Not B or M')
        featureVec, featuresUsed = [], []
        for i in range(2, len(split)):
            if possibleFeatures[i-2] not in dontUse:
                featureVec.append(float(split[i]))
                featuresUsed.append(possibleFeatures[i-2])
        data.append(Tumor(idNum, malignant, featuresUsed, featureVec))
    return data, featuresUsed


def readData(file_name):
    """
    Reads the data at file file_name into a list of
    strings, each encoding a data point

    file_name (string): name of the data file

    Returns a list of strings, each string encoding
            the ID, label, and features of a tumor
    """
    f = open(file_name)
    lines = f.readlines()
    data = [line.strip() for line in lines]
    f.close()
    return data
    
##### End of helper code


######################
##      PART 1      ##
######################

DATA = readData(FILENAME)

def splitData(data, holdout_frac):
    """
    Split data set into training and test sets
    
    data (list): list of elements
    holdout_frac (float): fraction of data points for testing,
                          as a float in [0,1] inclusive
    
    Returns a tuple (trainingData, testData) where trainingData and
            testData are both lists of elements and together
            partition the original data list
    
    Randomly split the data into training and test sets, each represented
    as a list, such that the test set takes roughly holdout_frac fraction
    of the original data
    Note: the implementation should be oblivious to the type of elements in the data list
    """
    testdata = []
    trainingdata = []
    
    for i in data: #iterates through each object in the data
        x = random.random() 
        if x <= holdout_frac: #determines if the datum is testing or training depending on the holdout fraction
            testdata.append(i) #appends to test data
        else:
            trainingdata.append(i) #appends to training data
    
    return (trainingdata, testdata)

######################
##      PART 2      ##
######################

def trainModel(train_set):
    """
    Trains a logistic regression model with the given dataset

    train_set (list): list of data points of type Tumor

    Returns a model of type sklearn.linear_model.LogisticRegression
            fit to the training data

    Hint: use the method fit(X,y) from the LogisticRegression class
    """
    features = []
    labels = []
    
    for i in train_set: #creates matrix of the features and a matrix of the labels
        features.append(i.getFeatures()) #adds features for each tumor in the training set
        labels.append(i.getLabel()) #adds a label for each tumor in the training set
    
    model = LogisticRegression() #builds the logistic regression
    fit = model.fit(features, labels) #creates the fit
    
    return fit

def predictLabels(model, threshold, data_points):
    """
    Uses the model and probability threshold to predict labels for
    the given data points

    model (LogisticRegression): a trained model
    threshold (float): a value between 0 and 1 to be used as a decision threshold
    data_points (list): list of Tumor objects for which to predict the labels

    Returns a list of labels (value 0 or 1), one for each data point

    Hint: use the method predict_proba(X) from the LogisticRegression class
    """
    features = []
    predicted = []
    
    for i in data_points: #builds a matrix of features for each tumor in the data points
        features.append(i.getFeatures())
    
    probmatrix = model.predict_proba(features) #generates the probability matrix
 
    for j in probmatrix: #for each entry in the probability matrix, it determines if the tumor is benign or malignant 
        if j[1] > threshold: #if the probability of the tumor being malignant is higher than the specified threshold, it is labeled as malignant
            predicted.append(1)
        else: #if the probability of the tumor being malignant is lower, then the tumor is modeled as benign
            predicted.append(0)
    
    return predicted

######################
##      PART 3      ##
######################

def scoreTestSet(model, threshold, test_set):
    """
    Uses the model and threshold to predict labels for the given data points,
    and compares the predicted labels to the true labels of the data.

    model (LogisticRegression): a trained model
    threshold (float): a value between 0 and 1 to be used as a decision threshold
    test_set (list): list of labeled Tumor objects to evaluate the model on

    Returns a tuple with the true positive, false positive, true negative, and
            false negative counts in that order
    """
    predicted = predictLabels(model, threshold, test_set) #the predicted labels for each tumor
    
    labels = []
    
    for i in test_set: #builds the list of labels for each tumor
        labels.append(i.getLabel())
    
    truep = 0 #empty true/false positive/negative values
    falsep = 0
    truen = 0
    falsen = 0
    
    for k in xrange(len(predicted)): #iterates through for each tumor
        if predicted[k] == 1:
            if predicted[k] == labels[k]: #if the predicted value is 1 and the same as the true value, its a true positive
                truep += 1
            else:
                falsep += 1 #otherwise the value is a false positive
        elif predicted[k] == 0:
            if predicted[k] == labels[k]: #if the predicted value is 0, and is the same as the labels, its a true negative 
                truen += 1
            else:
                falsen += 1 #otherwise the value is a false negative
        else:
            pass
    
    return (truep, falsep, truen, falsen)

def accuracy(truePos, falsePos, trueNeg, falseNeg):
    """
    Fraction of correctly identified elements
    
    truePos (int): number of true positive elements
    falsePos (int): number of false positive elements
    trueNeg (int): number of true negative elements
    falseNeg (int): number of false negative elements
    
    Returns the fraction of true positive or negative elements 
            out of all elements
    """
    x = truePos + trueNeg
    y = truePos + falsePos + trueNeg + falseNeg
    return float(x) / y #calculates the truepositive + truenegative over all values
           
def sensitivity(truePos, falseNeg):
    """
    Fraction of correctly identified positive elements out of all positive elements
    
    truePos (int): number of true positive elements
    falseNeg (int): number of false negative elements
    
    Returns the fraction of true positive elements out of all positive elements
    If there are no positive elements, returns a nan
    """
    x = truePos
    y = truePos + falseNeg    
    
    if y == 0:
        return float('nan') #if it divides by zero, it returns nan
    else:
        return float(x) / y #returns the true positives over all positives

def specificity(trueNeg, falsePos):
    """
    Fraction of correctly identified negative elements out of all negative elements

    trueNeg (int): number of true negative elements
    falsePos (int): number of false positive elements  
    
    Returns the fraction of true negative elements out of all negative elements
    If there are no negative elements, returns a nan
    """
    x = trueNeg
    y = trueNeg + falsePos
    
    if y == 0:
        return float('nan') #if it divides by zero, it returns nan
    else:
        return float(x) / y #returns the true negatives, over all negatives
    
def posPredVal(truePos, falsePos):
    """
    fraction of correctly identified positive elements 
    out of all positively identified elements
    
    truePos (int): number of true positive elements
    falsePos (int): number of false positive elements
    
    Returns the fraction of correctly identified positive elements 
            out of all positively identified elements  
    If no elements were identified as positive, returns a nan
    """
    x = truePos
    y = truePos + falsePos    
    
    if y == 0:
        return float('nan') #if it divides by zero, it returns nan
    else:
        return float(x) / y #divides the true positives over all positives

def buildROC(model, eval_data):
    """
    Plots the ROC curve, namely the true positive rate (y-axis) vs the false positive rate (x-axis)
    
    model (LogisticRegression): a trained logistic regression model
    eval_data (list): a list of Tumor instances on which to evaluate the model
    
    Returns the area under the curve
    
    Plot the ROC curve as measured in p values in the interval [0,1], inclusive, 
    in 0.01 increments
    At each p value, apply the model and measure the true positive rate 
    and false positive rate.
    
    Remember to give a meaningful title to your plot and include 
    in it the area under the curve 
    """
    p = 0
    trueprate = []
    falseprate = []
    
    while p <= 1:
        values = scoreTestSet(model, p, eval_data) #splits the data according to the p value
        falseprate.append(1 - specificity(values[2], values[1])) #calculates the false positive rate
        trueprate.append(sensitivity(values[0], values[3])) #calculates the true positive rate
        p += 0.01
    
    pylab.figure()    
    pylab.plot(falseprate, trueprate)
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    auc = sklearn.metrics.auc(falseprate, trueprate, reorder=True) #calculates the area under the curve
    pylab.title('False Positive vs True Positive where the Area Under the Curve is' + ' ' + str(auc))    
    
    return auc

def plotPerfVsCorruption(file_name, holdout_frac, threshold):
    """
    Plots model accuracy against the fraction of data that is corrupted

    file_name (string): the name of the file containing the uncorrupted data
    holdout_frac (float): fraction of data points for testing
    threshold (float): a value between 0 and 1 to be used as a decision threshold

    This function does not return anything

    Read the data from the specified file, using readData. For each value of r
    in the interval [0,1], inclusive, in 0.05 increments, generate a version of the data
    with r as the corruption fraction. For each corruption rate, split the corrupted data
    between training and testing sets using the specified holdout fraction, train a model
    on the training set and test on the testing set, using the threshold specified.
    Plot the model accuracy for each value of r
    """
    data = readData(file_name) #reads the data
    accuracies = [] #generates the needed empty values
    rval = []    
    r = 0.0
    
    while r <= 1: #creates the list of r values for the x axis 
        rval.append(r)
        r += 0.05    
    
    for i in rval: #for each r value:
        corruptdata = corruptData(data, i) #corrupts the data
        tumordata = getTumorData(corruptdata) #generates the list of tumor objects
        splitdata = splitData(tumordata[0], holdout_frac) #splits the data in accordance to the holdout fraction
        model = trainModel(splitdata[0]) #builds the model
        score = scoreTestSet(model, threshold, splitdata[1]) #creates the scores based on the model
        accuracies.append(accuracy(score[0], score[1], score[2], score[3])) #adds the calculated accuracy to the list of accuracies
    
    pylab.figure()
    pylab.plot(rval, accuracies)
    pylab.xlabel('Fraction of Corrupted Data')
    pylab.ylabel('Accuracy of Data')
    pylab.title('Fraction of Corrupted Data vs Accuracy')

######################
##      PART 4      ##
######################

def findBestFeatureToEliminate(training_data, features_to_omit, threshold):
    """
    Identifies the feature that most improves accuracy when removed. If no
    feature elimination improves over the accuracy using the entire set, then
    the returned feature is None

    training_data (list): a list of strings, each string encoding the
            ID, label, and features of a tumor
    features_to_omit (list): a list of strings, each the name of a feature that
            should be omited in the experiment
    threshold (float): a value between 0 and 1 to be used as a decision threshold

    Returns a tuple with the name of the best feature to eliminate and the best
            model. If no feature elimination results in improved accuracy, 
            the value returned as the feature name should be None, 
            and the model returned should be the model trained with 
            all the features

    Note A: the features in features_to_omit should not be used for training or
    evaluating any model. When we say 'model trained with all the features' we
    mean all features EXCEPT those in features_to_omit
    Note B: use the probability threshold specified for predicting labels
    Note C: use a holdout fraction of 0.2 for the development set

    Hint: getTumorData will prove very helpful
    """
    data = getTumorData(training_data, features_to_omit)[0] #builds the data with all omitted features
    
    splitdata = splitData(data, 0.2) #splits data
    model = trainModel(splitdata[0]) #builds model
    score = scoreTestSet(model, threshold, splitdata[1]) #calculates all of the scores for the dataset
   
    bestacc = accuracy(score[0], score[1], score[2], score[3]) #calculates the accuracy for the set with no extra removed features
    bestmodel = model #saves the model
    feature = None #best feature
    
    newomissions = features_to_omit[:] #copies the features to omit
    tumor = data[0] #selects an arbitrary tumor
    possiblefeatures = tumor.getFeatureNames() #finds the features present in that tumor
    
    for i in possiblefeatures: #iterates through all possible tumors
        newomissions.append(i) #adds the feature to the omitted features
        newdata = getTumorData(training_data, newomissions)[0] #builds new data with a new omitted feature
        newsplit = splitData(newdata, 0.2) #splits the new data
        newmodel = trainModel(newsplit[0]) #builds a model from the new data
        newscore = scoreTestSet(newmodel, threshold, newsplit[1]) #recalculates the scores with the new data
        newacc = accuracy(newscore[0], newscore[1], newscore[2], newscore[3]) #calculates the new accuracy
        newomissions.remove(i) #removes the feature from the omitted features list
        
        if newacc >= bestacc: #compares the new accuracy to the previous one and determines if its larger
            bestacc = newacc #if the new accuracy is higher than the old accuracy, it updates all information with the better removed feature
            bestmodel = newmodel
            feature = i
        else:
            pass
    
    return (feature, bestmodel)

def buildReducedModel(training_data, threshold):
    """
    Greedily eliminates features until no performance improvement is gained
    from elimination, and returns the best performing model

    training_data (list): a list of strings, each string encoding the
            ID, label, and features of a tumor
    threshold (float): a value between 0 and 1 to be used as a decision threshold

    Returns a tuple with the model trained on the best performing subset 
            of features along with the final list of features to omit which
            was used when training the model
    """
    features_to_omit = [] #starts with an empty features to omit list
    bestmodel = 0 #starts with an empty model
    
    omitted = True #sets omitted to true
    while omitted: 
        omitted = False #immediatly changes omitted to false so the loop breaks if its not changed back
        eliminate = findBestFeatureToEliminate(training_data, features_to_omit, threshold)
        if eliminate[0] == None: #if the find best feature to eliminate function returns a None, the omitted variable isn't changed from False and the loop will break
            pass
        else:
            features_to_omit.append(eliminate[0]) #if the find best feature to eliminate function returns an actual feature, it adds it to the features to omit list
            bestmodel = eliminate[1] #updates the best mode
            omitted = True #changes omitted to true so the loop continues
    
    return (bestmodel, features_to_omit)
    

######################
##      PART 5      ##
######################

def runExperiment(file_name, corrupt_frac, holdout_frac, threshold):
    """
    Trains and evaluates a model using all the features, then trains
    an improved model using feature reduction and evaluates it.

    file_name (string): name of data file
    corrupt_frac (float): fraction of data to be corrupted
    holdout_frac (float): fraction of data to be held out for testing
    threshold (float): a value between 0 and 1 to be used as a decision threshold

    Returns a tuple with the following values, in order:
            the accuracy of the full model evaluated on the training set,
            the accuracy of the full model evaluated on the testing set,
            the accuracy of the reduced model evaluated on the training set,
            and the accuracy of the reduced model evaluated on the testing set
    
    Note: your code should also generate plots and evaluation metrics as 
          discussed in the problem description
    """
    
    #corrupting the data
    uncorrupted_data = readData(file_name) #reads the data from the file
    corrupted_data = corruptData(uncorrupted_data, corrupt_frac) #corrupts the data in accordance to the corrupt fraction
    #------------
    
    #organizing the data and creating the model for the full set
    data_full = getTumorData(corrupted_data)[0] #generates and specifies the list of tumor objects
    split_data_full = splitData(data_full, holdout_frac) #splits the data into training and test in accordance to the holdout fraction
    full_model = trainModel(split_data_full[0]) #builds the model from the training data
    #------------
    
    #organizing the data and creating a model for the reduced set
    reduced = buildReducedModel(corrupted_data, threshold) #Finds the best reduced model and omitted features
    reduced_model = reduced[0] #specifies the reduced model
    omitted_features = reduced[1] #specifies the omitted features
    reduced_data = getTumorData(corrupted_data, omitted_features)[0] #builds and specifies the list of tumor objects
    split_data_reduced = splitData(reduced_data, holdout_frac) #splits the data in accordance to the holdout fraction
    #------------
    
    #Scoring the full data
    full_score_training = scoreTestSet(full_model, threshold, split_data_full[0]) #scores the full training data
    full_score_testing = scoreTestSet(full_model, threshold, split_data_full[1]) #scores the full testing data
    #-------------
    
    #scoring the reduced data
    reduced_score_training = scoreTestSet(reduced_model, threshold, split_data_reduced[0]) #scores the reduced training data
    reduced_score_testing = scoreTestSet(reduced_model, threshold, split_data_reduced[1]) #scores the reduced testing data
    #-------------
    
    #determining accuracy of full data
    full_training_set = accuracy(full_score_training[0], full_score_training[1], full_score_training[2], full_score_training[3]) #finds the accuracy of the full training set
    full_testing_set =  accuracy(full_score_testing[0], full_score_testing[1], full_score_testing[2], full_score_testing[3]) #finds the accuracy of the full testing set
    #-------------
    
    #determining accuracy of reduced data
    reduced_training_set = accuracy(reduced_score_training[0], reduced_score_training[1], reduced_score_training[2], reduced_score_training[3]) #finds the accuracy of the reduced training set
    reduced_testing_set = accuracy(reduced_score_testing[0], reduced_score_testing[1], reduced_score_testing[2], reduced_score_testing[3]) #finds the accuracy of the reduced testing set
    
    #plotting the relevant plots
    buildROC(full_model, split_data_full[1]) #plots the ROC
    plotPerfVsCorruption(file_name, holdout_frac, threshold) #plots corruption vs accuracy
    #------------
    
    return (full_training_set, full_testing_set, reduced_training_set, reduced_testing_set)
    

runExperiment(FILENAME, DEFAULT_CORRUPT_FRAC, DEFAULT_HOLDOUT_FRAC, DEFAULT_THRESHOLD)


