#============================================================================

#    CS379 – Unit 5 -Submission Node 5– Applied Machine Learning

#    Filename: Unit 5 Submission Node: my-unit5-submission-node5.doc

#    Author: Shelerina N Hawkes Date: September 2, 2021

#    Purpose: Demonstrate Python programming in the Spyder

#    development environment by creating a program that uses the decision tree

#    implementation on the Titanic dataset.

#============================================================================
#Import necessary modules
import numpy as np
import os
import pandas as pd
import pydotplus
from IPython.display import Image
from six import StringIO
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

#Function to count nulls       
def list_nulls(df):
    feat_list = list(df.columns.values)
    for feat in feat_list:
        print (feat,": ",sum(pd.isnull(df[feat])))
        
#Function to run the decsion tree
def run_model(criterion_var, splitter_var):       
    dt = DecisionTreeClassifier(criterion=criterion_var,splitter=splitter_var)
    dt.fit(X_train, y_train)#Train the classifier on training data
    y_pred = dt.predict(X_test)#Apply trained classifier to X_test    
    print('Accuracy score:',accuracy_score(y_test, y_pred)*100) 
    print('Confusion matrix:\n',confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return(dt)

#Setting file path
# os.getcwd()
# wd = 'C:\\Users\\shawk\\.spyder-py3'
# os.chdir(wd)

df = pd.read_csv("Titanic_IP5.csv")#Reading in the dataset

df.info()#Inspecting dataframe to determine necessary columns

list_nulls(df)#Showing columns with nulls

#Remove irrelevant columns from new dataframe
new_df = df.drop(["name","cabin","body","embarked","ticket","boat",
                         "home.dest"], axis = 1, inplace = False)
new_df.info()
list_nulls(new_df)               

#Replace any missing values with median 
new_df.fillna(new_df.median(), inplace = True)

#Convert sex to numerical value
le = preprocessing.LabelEncoder() 
le.fit(df['sex']) 
list(le.classes_)
new_df['sex'] = le.transform(new_df['sex'])

#Copy everything but survived to X. Copy only survived to y
X = pd.DataFrame(new_df.drop(['survived'], 1).astype(float))
y = np.array(new_df['survived']).astype(int)

#Split data into 70% train and 30% test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#Scale data for uniformity
scaler = StandardScaler()
scaler.fit(X_train)

#Determine best tree
dt = run_model('gini','best')
dt = run_model('gini','random')
dt = run_model('entropy','best')
dt = run_model('entropy','random')

#Select features for graph
feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
decision_tree_data = StringIO()
export_graphviz(dt,out_file = decision_tree_data, filled = True,rounded = True,
                special_characters = True,feature_names = feature_cols,
                class_names = ['0','1'])

#Print tree & save it to file
graph = pydotplus.graph_from_dot_data(decision_tree_data.getvalue())
graph.write_png('decsion_tree_results.png')
Image(graph.create_png())


