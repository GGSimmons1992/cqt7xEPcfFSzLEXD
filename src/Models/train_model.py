#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import math
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.neighbors import KNeighborsClassifier as knn
from xgboost import XGBClassifier as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as gnb
import sklearn.model_selection as ms
import pickle
import sys
sys.path.insert(0, "../Data/")
from make_dataset import loadData


# In[2]:


def retrieveModelsBasedOnModelType(modelType):
    if modelType == 'tree':
        gridmodel = tree(random_state=51)
        finalmodel = tree(random_state=51)
    elif modelType == 'forest':
        gridmodel = rf(random_state=51)
        finalmodel = rf(random_state=51)
    elif modelType == 'knn':
        gridmodel = knn()
        finalmodel = knn()
    elif modelType == 'xgboost':
        gridmodel = xgb(random_state=51)
        finalmodel = xgb(random_state=51)
    elif modelType == 'svm':
        gridmodel = SVC(random_state=51)
        finalmodel = SVC(random_state=51)
    else:
        raise Exception("modelType Value not considered. Please choose from ['tree','forest','knn','xgboost','svm']")
    return gridmodel,finalmodel


# In[3]:


def fitModelWithGridSearch(searchParams,XTrain,yTrain,modelType):
    gridmodel,finalmodel = retrieveModelsBasedOnModelType(modelType)
    modelGridSearch = ms.GridSearchCV(gridmodel, param_grid=searchParams,scoring='accuracy',cv=6)
    modelGridSearch.fit(XTrain,yTrain)
    finalmodel.set_params(**modelGridSearch.best_params_)
    return finalmodel

# In[6]:


def printScore(model,X,y,dataSetType):
    print(f"{dataSetType} accruacy: {model.score(X,y)}")


# In[7]:


def saveModel(model,modelName):
    pickle.dump(model, open(f"../../Models/{modelName}.pkl", 'wb'))


# In[8]:


def main():
    np.random.seed(51)
    XTrain,XTest,yTrain,yTest = loadData()
    
    nRows = XTrain.shape[0]
    sqrtNRows = int(math.sqrt(nRows))
    log2NRows = int(math.log2(nRows))
    possibleThirdGeometricTerm1 = int((sqrtNRows ** 2)/log2NRows)
    possibleThirdGeometricTerm2 = int((log2NRows ** 2)/sqrtNRows)
    suggestedMaxKRange = [possibleThirdGeometricTerm1,possibleThirdGeometricTerm2]
    kRange = [int(x) for x in np.linspace(5,max(suggestedMaxKRange),10)]
    
    treeParams = {
        "max_depth":[2,3],
        "max_features":[2,3],
        "criterion": ["gini","entropy"]
    }
    forestParams = {
        "n_estimators": [100,150,200,250,300],
        "max_depth":[2,3],
        "max_features":[2,3],
        "criterion": ["gini","entropy"]
    }
    knnParams = {
        "n_neighbors": kRange
    }
    xgbParams = {
        "learning_rate": list(np.linspace(.1,1,10))
    }
    svmParams = {
        "kernel": ["linear","rbf","poly","sigmoid"],
        "gamma": ["auto","scale"]
    }
    
    logModel = lm.LogisticRegression(max_iter=1e9)
    logPipe = make_pipeline(StandardScaler(), logModel)
    gnbModel = gnb()
    
    estimators = [
        ("logModel",logPipe),
        ("naiveBayes",gnbModel),
        ("tree",fitModelWithGridSearch(treeParams,XTrain,yTrain,'tree')),
        ("forest",fitModelWithGridSearch(forestParams,XTrain,yTrain,'forest')),
        ("knn",fitModelWithGridSearch(knnParams,XTrain,yTrain,'knn')),
        ("xgboost",fitModelWithGridSearch(xgbParams,XTrain,yTrain,'xgboost')),
        ("svm",fitModelWithGridSearch(svmParams,XTrain,yTrain,'svm'))
    ]
    
    goodModels = []
    
    for est in estimators:
        modName = est[0]
        mod = est[1]
        mod.fit(XTrain,yTrain)
        print(modName)
        printScore(mod,XTrain,yTrain,"Training")
        printScore(mod,XTest,yTest,"Testing")
        if mod.score(XTest,yTest) > 0.73:
            saveModel(mod,modName)
            goodModels.append(modName)
    
    goodModelsDictionary = {
        "goodModels": goodModels
    }
    
    with open('../../Models/goodModelsDictionary.json', 'w') as fp:
        json.dump(goodModelsDictionary, fp)


# In[9]:

if __name__=='__main__':
    main()




