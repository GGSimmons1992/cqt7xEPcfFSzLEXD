import pickle
import json
import sys
import pandas as pd

newInputData = f"../../Data/External/{sys.argv[1]}.csv"

df = pd.read_csv(newInputData)

with open('../../Models/goodModelsDictionary.json') as d:
    goodModels=json.load(d)["goodModels"]

for modName in goodModels:
    print(modName)
    mod = pickle.load(open(f"../../Models/{modName}.pkl", 'rb'))
    prediction = mod.predict(df)
    print(prediction)
    print("---")
        

    



