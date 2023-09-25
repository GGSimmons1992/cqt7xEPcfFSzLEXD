import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.tree import DecisionTreeClassifier as tree
import sklearn.model_selection as ms
import sklearn.metrics as sm

def displayFeatureImportances(columns,fittedModel,modelName):
    importance = fittedModel.feature_importances_
    featureImportance = pd.DataFrame({
        "feature": columns,
        "featureImportance": importance
        },columns = ["feature","featureImportance"])
    featureImportancesSorted = featureImportance.sort_values(by="featureImportance", ascending=False)
    print(f'{modelName} top 10 feature importances')
    for i in range(len(columns)):
        featureRow = featureImportancesSorted.iloc[i]
        feature = featureRow['feature']
        featureValue = featureRow['featureImportance']
        print(f'Rank {i}: {feature}: score: {featureValue}')
    print("\n")

def fitForestWithGridSearch(searchParams,XTrain,yTrain):
    modelGridSearch = ms.GridSearchCV(rf(), param_grid=searchParams)
    modelGridSearch.fit(XTrain,yTrain)
    modelParams = modelGridSearch.best_params_
    print(modelParams)
    model = rf(n_estimators = modelParams["n_estimators"],
                max_depth = modelParams["max_depth"],
                max_features = modelParams["max_features"])
    model.fit(XTrain,yTrain)
    yPredict = model.predict(XTrain)
    print(f"Training Score {sm.accuracy_score(yTrain,yPredict)}")
    return model

def fitTreeWithGridSearch(searchParams,XTrain,yTrain):
    modelGridSearch = ms.GridSearchCV(rf(), param_grid=searchParams)
    modelGridSearch.fit(XTrain,yTrain)
    modelParams = modelGridSearch.best_params_
    print(modelParams)
    model = tree(max_depth = modelParams["max_depth"],
                max_features = modelParams["max_features"])
    model.fit(XTrain,yTrain)
    yPredict = model.predict(XTrain)
    print(f"Training Score {sm.accuracy_score(yTrain,yPredict)}")
    return model