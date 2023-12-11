import pickle
import json
import sys
import sys
sys.path.insert(0, "../Data/")
import pandas as pd
import numpy as np
import process_data as prc

newInputData = f"../../Data/External/{sys.argv[1]}.csv"
def main():
    df = pd.read_csv(newInputData)

    with open('../../Models/goodModelsDictionary.json') as d:
        gmDict = json.load(d)
    goodModels = gmDict["goodModels"]
    goodModelScores = gmDict["goodModelScores"]
    modName = goodModels[np.argmax(goodModelScores)]
    if 'Under' in modName:
        combinedName = 'TermDepositUnder'
    elif 'Over' in modName:
        combinedName = 'TermDepositOver'
    elif 'NearMiss' in modName:
        combinedName = 'TermDepositNearMiss'
    else:
        combinedName = 'TermDepositSmote'

    numericalDF = prc.removeUnimportantNumericalColumns(df,None,combinedName,"test")
    categoricalDF = prc.removeUnimportantCategoricalColumns(df,None,combinedName,"test")
    scaledDF = prc.scaleTestDF(numericalDF,combinedName)
    encodedDF = prc.encodeTestDF(categoricalDF,combinedName)
    df = pd.concat([scaledDF,encodedDF],axis=1)

    categoricalColumns = pickle.load(open(f'../../Data/Interim/{combinedName}SignificantCategoricalCols.pkl','rb'))
    oheColumns = pickle.load(open(f'../../Data/Interim/{combinedName}OheColumns.pkl','rb'))

    if ('xgboost' in modName) | ('tree' in modName) | ('forest' in modName):
        df = df.drop(oheColumns,axis=1)
    else:
        df = df.drop(categoricalColumns,axis=1)

    mod = pickle.load(open(f"../../Models/{modName}.pkl", 'rb'))
    prediction = mod.predict(df)
    print(prediction)
    
if __name__=='__main__':
    main()    

    



