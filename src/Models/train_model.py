#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore", message="is_sparse is deprecated and will be removed in a future version.")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import json
import math
from sklearn.ensemble import RandomForestClassifier as rf
import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.neighbors import KNeighborsClassifier as knn
from xgboost import XGBClassifier as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as gnb
import sklearn.model_selection as ms
import sklearn.metrics as sm
import pickle



# In[2]:


def retrieveModelsBasedOnModelType(modelType):
    if modelType == 'log':
        gridmodel = lm.LogisticRegression(random_state=51,penalty='l2',C=0.001)
    elif modelType == 'naiveBayes':
        gridmodel = gnb()
    elif modelType == 'tree':
        gridmodel = tree(random_state=51)
    elif modelType == 'forest':
        gridmodel = rf(random_state=51, oob_score=True)
    elif modelType == 'knn':
        gridmodel = knn()
    elif modelType == 'xgboost':
        gridmodel = xgb(random_state=51,reg_alpha=1000,reg_lambda=1000)
    elif modelType == 'svm':
        gridmodel = SVC(random_state=51)
    else:
        raise Exception("modelType Value not considered. Please choose from ['log','naiveBayes','tree','forest','knn','xgboost','svm']")
    return gridmodel


# In[3]:


def fitModelWithGridSearch(searchParams,XTrain,yTrain,modelType):
    gridmodel = retrieveModelsBasedOnModelType(modelType)
    modelGridSearch = ms.GridSearchCV(gridmodel, param_grid=searchParams,scoring='f1',
                                      cv=ms.StratifiedKFold(n_splits=5,random_state=51, shuffle=True),n_jobs=-1)
    modelGridSearch.fit(XTrain,yTrain)
    return modelGridSearch


# In[4]:


def processPredictData(predictFileName):
    df = pd.read_csv(f"../../Data/External/{predictFileName}.csv")
    scaledDF = scaleTestData(df)
    encodedDF = transformDF(scaledDF)
    encodedDF.to_csv(f'../../Data/External/{predictFileName}Final.csv',index=False)


# In[5]:


def loadData(dataType,baseName):
    TermDepositData = None
    if dataType == "train":
        TermDepositData = pd.read_csv(f"../../Data/Processed/{baseName}Train.csv")
    else:
        TermDepositData = pd.read_csv(f"../../Data/Processed/{baseName}Test.csv")
    y = TermDepositData[["y"]].values.ravel()
    X = TermDepositData.drop("y",axis=1)
    return X,y


# In[6]:


def getTreeFeatureRange(baseName):
    fullName = baseName+'Under'
    XTrainOriginal,yTrain = loadData("train",fullName)
    oheColumns = pickle.load(open(f'../../Data/Interim/{fullName}OheColumns.pkl','rb'))
    nTreeCols = XTrainOriginal.shape[1] - len(oheColumns)
    twoEights = int(2*nTreeCols/8)
    fiveEights = int(5*nTreeCols/8)
    return [twoEights,fiveEights]


# In[7]:


def printScore(trueY,predictY,dataSetType):
    scoreValue = sm.f1_score(trueY,predictY)
    print(f"{dataSetType} report")
    print(sm.classification_report(trueY,predictY))
    return scoreValue


# In[8]:


def saveModel(model,modelName):
    pickle.dump(model, open(f"../../Models/{modelName}.pkl", 'wb'))


# In[9]:


def main():
    baseName = "TermDeposit"
    np.random.seed(51)
    
    """
    models = [
        (LogisticRegression, {'max_iter': [1000, 1500, 2000]}),
        (KNeighborsClassifier, {'n_neighbors': np.arange(2, 10, 1)}),
        (DecisionTreeClassifier, {'max_depth': np.arange(5, 10, 1)}),
        (RandomForestClassifier, {'n_estimators': np.arange(5, 10, 1)}),
        (xgb.XGBClassifier, {'n_estimators': [100, 150, 200], 'subsample': [0.8, 0.9, 1]})
    ]
    """
    logParams = {
        'max_iter': [1000, 1500, 2000]
    }
    
    bayesParams = {
        "var_smoothing": [1e-3,1e-6,1e-9]
    }
    
    treeParams = {
        'max_depth': np.arange(5, 10, 1)
    }
    forestParams = {
        'n_estimators': np.arange(5, 10, 1)
    }
    xgbParams = {
        'n_estimators': [100, 150, 200], 
        'subsample': [0.8, 0.9, 1]
    }
    svmParams = {
        "kernel": ["linear","rbf","poly","sigmoid"],
        "gamma": ["auto","scale"],
        "max_iter": [1000, 1500, 2000],
        "class_weight": ["balanced"],
        "probability": [True]
    }
    
    knnParams = {
        "n_neighbors": np.arange(2, 10, 1)
    }
    
    goodModels = []
    goodModelScores = []
    
    for balanceType in ["Under","Over","Smote","NearMiss"]:
        fullName = baseName + balanceType
        categoricalColumns = pickle.load(open(f'../../Data/Interim/{fullName}SignificantCategoricalCols.pkl','rb'))
        oheColumns = pickle.load(open(f'../../Data/Interim/{fullName}OheColumns.pkl','rb'))
        
        XTrainOriginal,yTrain = loadData("train",fullName)
        XTestOriginal,yTest = loadData("test",fullName)
        XTrainOHE = XTrainOriginal.drop(categoricalColumns,axis=1)
        XTestOHE = XTestOriginal.drop(categoricalColumns,axis=1)
        XTrainLE = XTrainOriginal.drop(oheColumns,axis=1)
        XTestLE = XTestOriginal.drop(oheColumns,axis=1)
    
        estimators = [
            ("logModel",fitModelWithGridSearch(logParams,XTrainOHE,yTrain,'log'),'onehot'),
            ("naiveBayes",fitModelWithGridSearch(bayesParams,XTrainOHE,yTrain,'naiveBayes'),'onehot'),
            ("tree",fitModelWithGridSearch(treeParams,XTrainLE,yTrain,'tree'),'label'),
            ("forest",fitModelWithGridSearch(forestParams,XTrainLE,yTrain,'forest'),'label'),
            ("knn",fitModelWithGridSearch(knnParams,XTrainOHE,yTrain,'knn'),'onehot'),
            ("xgboost",fitModelWithGridSearch(xgbParams,XTrainLE,yTrain,'xgboost'),'label'),
            ("svm",fitModelWithGridSearch(svmParams,XTrainOHE,yTrain,'svm'),'onehot')
        ]
        
        for est in estimators:
            modName = f'{est[0]}{balanceType}sample'
            displayName = f'{est[0]} {balanceType}sample'
            mod = est[1]
            if est[2] == 'onehot':
                XTrain,XTest = XTrainOHE,XTestOHE
            else:
                XTrain,XTest = XTrainLE,XTestLE
                
            predictTrainY = mod.predict(XTrain)
            predictTestY = mod.predict(XTest)
            print(displayName)
            print(mod.best_estimator_.get_params())
            trainScore = printScore(yTrain,predictTrainY,"Training")
            testScore = printScore(yTest,predictTestY,"Testing")
            if testScore > 0.81:
                saveModel(mod,modName)
                goodModels.append(modName)
                goodModelScores.append(testScore)

    goodModelsDictionary = {
        "goodModels": goodModels,
        "goodModelScores": goodModelScores
    }

    with open('../../Models/goodModelsDictionary.json', 'w') as fp:
        json.dump(goodModelsDictionary, fp)


# In[10]:


if __name__=='__main__':
    main()


# In[ ]:




