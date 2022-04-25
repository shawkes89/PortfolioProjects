#======================================================================

#    Author: Shelerina N Hawkes Date: August 18, 2021

#    Purpose: Demonstrate Python programming in the Spyder

#    development environment by creating a program that demonstrates

#    streaming data and machine learning algorithms.

#=======================================================================
#import necessary modules. All functions are listed on top for uniformity
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tweepy as tw
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report

#Functions to count nulls      
def list_nulls(df):
    feat_list = list(df.columns.values)
    for feat in feat_list:
        print (feat,": ",sum(pd.isnull(df[feat])))

#Calculate y and pred_y to get percentage         
def get_accuracy(y, pred_y):
    correct = 0
    for i in range(len(y)):
        if pred_y[i] == y[i]:
            correct += 1 
            
    print(correct/len(X))
    
#Function to transform columns  
def do_labelencode(df,feat):
    le.fit(df[feat]) 
    df[feat] = le.transform(df[feat])
    
#Kmeans algorithm 
def run_model(X, num_clusters):
    kmeans = KMeans(n_clusters = num_clusters, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    pred_y = kmeans.predict(X)
    
    print(metrics.accuracy_score(y, pred_y))

#Authenticate to Twitter
auth = tw.OAuthHandler("", 
                       "")
auth.set_access_token("", 
                      "")

#Create API object
api = tw.API(auth, wait_on_rate_limit = True,
    wait_on_rate_limit_notify = True)
#sleep_on_rate_limit=False

#Verify credentials
try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")
    
search_words = "#COVID19"
date_since = "2021-08-01"

#Collect a list of tweets
tweets = tw.Cursor(api.search,
              q=search_words,
              since=date_since).items(1000)

tweets_data = [[tweet.created_at, tweet.user.screen_name, tweet.user.location,
                tweet.favorited,tweet.retweet_count,
                tweet.favorite_count,tweet.lang, 
                tweet.is_quote_status,tweet.retweeted]for tweet in tweets]

tweets_data

df = pd.DataFrame(data=tweets_data,
                    columns=['created_at','user', "location",'favorited',
                              'retweet_count','favorite_count','lang',
                              'is_quote_status','retweeted'])

df.info()#Inspecting dataframe to determine necessary columns
list_nulls(df)#Showing columns with nulls, if any

#Double check if location is null
sum(pd.isnull(df['location']))

#Drop irrelevant columns
df.drop(["location"], axis = 1, inplace = True)

#Shows state of data prior to predictions
g = sns.FacetGrid(df, col ='is_quote_status')
g.map(plt.hist, 'retweet_count')

#Convert to numerical value
le = preprocessing.LabelEncoder() 
do_labelencode(df,'user')
do_labelencode(df,'favorited')
do_labelencode(df,'lang')
do_labelencode(df,'is_quote_status')
do_labelencode(df,'retweeted')

df_clean = df

store = pd.HDFStore('df_clean.h5')
store['df_clean'] = df_clean  #Save dataframe
# df_clean = store['df_clean']  #Load dataframe
# df = df_clean

X = df_clean

#Copy everything but created_at and is_quote_status to X. 
#Copy only is_quote_status to y
X = pd.DataFrame(df.drop(['created_at','favorited'], 1).astype(float))
y = np.array(df['favorited']).astype(int)
   
#Determine number of ideal clusters by measuring performance metric
run_model(X,2)
run_model(X,3)
run_model(X,4)
run_model(X,5)
run_model(X,6)

#Runs Kmeans alone with local var pred_y using optimal clusters
kmeans = KMeans(n_clusters = 2, init='k-means++', max_iter = 300, n_init = 10, 
                random_state = 0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

#Evaluate the K means algorithm  
cf = confusion_matrix(y, y_pred)
print(confusion_matrix)
print(classification_report(y, y_pred))
print(accuracy_score(y, y_pred))

#Shows state of data after predictions
g = sns.FacetGrid(df, col ='favorited')
g.map(plt.hist, 'retweet_count')

