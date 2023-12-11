# Term Deposit Marketing

## Background 
This is a classification project making a product recommendations to customers from a mock dataset. This particular dataset is skewed mostly towards no, thus compelling the modeler to balance the dataset with random undersampling, random oversampling, SMOTE, and NearMiss algorithms. This challenge also encouraged scaling, utilizing one-hot encoding and label encoding, and feature selection through chi square and spearmanr.

Data Description:

y : target attribute (y) with values indicating yes or no to recommend to customers
age : age of customer (numeric)
job : type of job (categorical)
marital : marital status (categorical)
education : (categorical)
default: has credit in default? (binary)
balance: average yearly balance, in euros (numeric)
housing: has a housing loan? (binary)
loan: has personal loan? (binary)
contact: contact communication type (categorical)
day: last contact day of the month (numeric)
month: last contact month of year (categorical)
duration: last contact duration, in seconds (numeric)
campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
Using the the dataset provided by Apziva, and the key above, the goal is to reach an F1 score. External holds feature input data that will be used for predictions

## Data
Data is not pushed due to gitignore. Raw data from Apziva is in Raw folder. Running process_data.py in src/Data will insert transformed train and test data into Processed folder. 
## Models
goodModelsDictionary.json has the names and f1 scores of the pickle files of the models that passed. 
## Notebooks
eda.ipynb is an eda exploration of the data set. The process_data.py methods and script originates from makeData.ipynb, and train_model.py methods and script originates from train.ipynb
## src
 __init__.py is a boilerplate that could be used for future models; it's just a placeholder for this repo. Models/process_data.py has methods to load train and test sets through loadData() or created train and test sets through makeData(). Running Models/train_model.py trains and evaluates the models and stores the passing models in root/Models folder. Running Models/predict_model will make predictions on a features only data set in Data/External (i.e. running "python predict_model.py mockData" will predict on the input dataset Data/External/mockData.csv and print the predictions).
## Requirements.txt
list of python packages used by this repo.
## Conclusion
Of the different attempted models, 16 models were able to achieve above 81% F1 score, with the best score going to the oversampled forest.
