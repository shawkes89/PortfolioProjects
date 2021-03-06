#========================================================================

#    Author: Shelerina N Hawkes Date: August 3, 2021

#    Purpose: Demonstrate Python programming in the Spyder

#    development environment by creating a program that demonstrates

#    the use of Kmeans clustering

#========================================================================
#import necessary modules. All functions are listed on top for uniformity 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans

#Functions to count nulls & calculate y and pred_y to get percentage       
def list_nulls(df):
    feat_list = list(df.columns.values)
    for feat in feat_list:
        print (feat,": ",sum(pd.isnull(df[feat])))
        
def get_accuracy(y, pred_y):
    correct = 0
    for i in range(len(y)):
        if pred_y[i] == y[i]:
            correct += 1    
    print(correct/len(X))

#Reading in the dataset
data_titanic = pd.read_csv("Titanic_IP1.csv")

#Inspecting dataframe to determine necessary columns
data_titanic.info()

#Showing columns with nulls
list_nulls(data_titanic)

#Remove irrelevant columns from new dataframe
df = data_titanic.drop(["name", "cabin","body","embarked","ticket","boat",
                        "home.dest"], axis = 1, inplace = False)
df.info()
list_nulls(df)

#Shows state of data prior to predictions
g = sns.FacetGrid(df, col ='survived')
g.map(plt.hist, 'sex', bins = 20)

grid = sns.FacetGrid(df, col ='survived', row ='pclass', height =2.2, aspect=1.6)
grid.map(plt.hist, 'age', bins = 20)
grid.add_legend();

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

#Kmeans algorithm 
def run_model(X, num_clusters):
    kmeans = KMeans(n_clusters = num_clusters, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    pred_y = kmeans.predict(X)
    get_accuracy(y, pred_y)

#Determine number of ideal clusters by measuring performance metric
run_model(X,2)
run_model(X,3)
run_model(X,4)
run_model(X,5)
run_model(X,6)

#Runs Kmeans alone with local var pred_y using optimal clusters
kmeans = KMeans(n_clusters = 3, init='k-means++', max_iter = 300, n_init = 10, 
                random_state = 0)
kmeans.fit(X)
pred_y = kmeans.predict(X)

#Separate and set the data frames next to eachother
df2 = df
df2['survived'] = y
df2['survived_kmeans'] = pred_y

#Shows state of data after predictions
g = sns.FacetGrid(df, col ='survived')
g.map(plt.hist, 'sex', bins = 20)

grid = sns.FacetGrid(df, col ='survived', row ='pclass', height =2.2, aspect=1.6)
grid.map(plt.hist, 'age', bins = 20)
grid.add_legend();




