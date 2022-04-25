#========================================================================

#    CS379 – Unit 1 -Submission Node 1– Introduction to Machine Learning

#    Filename: Unit 1 Submission Node: my-unit1-submission-node1.doc

#    Author: Shelerina N Hawkes Date: August 3, 2021

#    Purpose: Demonstrate Python programming in the Spyder

#    development environment by creating a program that demonstrates

#    the use of KNN classification algorithm

#========================================================================
#import necessary modules. All functions are listed on top for uniformity 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Function to count nulls  
def list_nulls(df):
    feat_list = list(df.columns.values)
    for feat in feat_list:
        print (feat,": ",sum(pd.isnull(df[feat])))
    
#Packages scaler and model together
#Uses cross_val_score to train 80% of data and test 20%. 
#Generate 5 scores as it tests 5 different times
#Line 43 shows how to split the data        
def run_model(X_train, X_test, y_train, y_test, num_neighbors):
    knn = KNeighborsClassifier(n_neighbors = num_neighbors, metric='euclidean')
    ss = StandardScaler() 
    pipeline = Pipeline([('transformer', ss), ('estimator', knn)])
    skf = StratifiedKFold(n_splits = 5, random_state = 30, shuffle = True) 
    scores = cross_val_score(pipeline, X, y, cv = skf)
    print(scores)
               
#Read in dataset
df = pd.read_csv("Titanic_IP2.csv")

#Inspecting dataframe to determine necessary columns
df.info()

#Showing columns with nulls
list_nulls(df)

#Remove irrelevant columns from new dataframe
df = df.drop(["name", "cabin","body","embarked","ticket","boat",
              "home.dest"], axis = 1, inplace = False)
df.info()
list_nulls(df)

#Replace any missing values with median 
df.fillna(df.median(), inplace = True)

#Convert sex to numerical value
le = preprocessing.LabelEncoder() 
le.fit(df['sex']) 
list(le.classes_)
df['sex'] = le.transform(df['sex'])

#Copy everything but survived to X. Copy only survived to y
X = np.array(df.drop(['survived'], 1).astype(float))
y = np.array(df['survived']).astype(int)

#Split data into 80% train and 20% test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Scale data for uniformity
scaler = StandardScaler()
scaler.fit(X_train)

#KNN model to help predict K value. 
k_range = range(1, 11)
score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score.append(metrics.accuracy_score(y_test,y_pred))
print(score)

#Determine which K value has highest accuracy 
plt.plot(k_range, score)
plt.xlabel("Value of K for KNN")
plt.ylabel("Accuracy Test")

#Training the model & predicting test results
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy percentage:",metrics.accuracy_score(y_test, y_pred))

#Run model to help see results
run_model(X_train, X_test, y_train, y_test, 2)
run_model(X_train, X_test, y_train, y_test, 3)
run_model(X_train, X_test, y_train, y_test, 4)
run_model(X_train, X_test, y_train, y_test, 5)






