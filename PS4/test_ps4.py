# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:30:21 2015

@author: Nico, Yonatan 
"""

DATA_FILE = 'tumorInfo.txt'

import ps4

print 'Testing splitData'
data, feats = ps4.getTumorData(ps4.readData(DATA_FILE))
train, test = ps4.splitData(data, 0.8)
assert len(train) + len(test) == len(data), 'splitData does not return a valid partition'
for x in train:
	assert x not in test, 'splitData does not return a valid partition: Tumor instance inluded in both train and test sets'
print 'SUCCESS: splitData passed tests'


print 'Testing performance metrics (problem 3.2)'
assert 0 == ps4.accuracy(0.0, 1.0, 0.0, 1.0), "accuracy(0,1,0,1) != 0"
assert 0 == ps4.sensitivity(0.0, 1.0), "sensitivity(0,1) != 0"
assert 0 == ps4.specificity(0.0, 1.0), "specificity(0,1) != 0"
assert 0 == ps4.posPredVal(0.0, 1.0), "posPredVal(0,1) != 0"
assert 0.5 == ps4.accuracy(1.0,1.0,1.0,1.0), "accuracy(1,1,1,1) != 0.5"
assert 0.5 == ps4.sensitivity(1.0,1.0), "sensitivity(1,1,1,1) != 0.5"
assert 0.5 == ps4.specificity(1.0,1.0), "specificity(1,1,1,1) != 0.5"
assert 0.5 == ps4.posPredVal(1.0,1.0), "posPredVal(1,1,1,1) != 0.5"
print 'SUCCESS: passed problem 3.2 tests'

