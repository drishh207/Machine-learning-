#database used: emails.csv from kaggle database
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import os
import math

#loading the dataset
emails=pd.read_csv('emails.csv',header='infer')
print(emails.shape)
emails.head()


out=None
for emails in pd.read_csv('emails.csv',header='infer',chunksize=1000):
    if out is None:
        out=emails.copy()
    else:
        out.append(emails)
    out.head()
    
#extract to,from and body from each email
def email_parsing(raw_message):
    lines=raw_message.split('\n')
    email={}
    message=''
    keys_to_extract=['from','to']
    for line in lines:
        if ':' not in line:
            message+=line.strip()
            email['body']=message
        else:
            pairs=line.split(':')
            key=pairs[0].lower()
            val=pairs[1].strip()
            if key in keys_to_extract:
                email[key]=val
    return email
    

#compile all extracted data into a single dictionary
def emails_parsing(messages):
    emails=[email_parsing(message) for message in messages]
    return {
        'body': result_append(emails,'body'),
        'to': result_append(emails,'to'),
        'from_': result_append(emails,'from')
        }

#append all key results into single result
def result_append(emails,key):
    results=[]
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results

#create a new dataframe for extracted data
extracted_data=pd.DataFrame(emails_parsing(out['message']))

#check any NaN values
extracted_data.isnull().any()

#converting categorical data into numerical form
stopwords=ENGLISH_STOP_WORDS.union(['ect','hou','com','recipient'])
vect=TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.1, min_df=2)

#email bodies are converted to document term matrix.
X=vect.fit_transform(extracted_data.body)

#To Visualise this matrix we need 2D representation of the DTM(Document term matrix)
X_dense=X.todense()
coords=PCA(n_components=2).fit_transform(X_dense)
plt.scatter(coords[:,0],coords[:,1],c='b')
plt.show()

#now top words from each email are to be found by creating a variable(features) that holds a list of all the words in the tf-idfs in the same order as the columns in tne matrix
features=vect.get_feature_names()

#This function top_tfidf_features takes a single row of the tf-idf matrix and returns the highest scoring words
def top_tfidf_features(row,features,top_words=25):
    top_ids=np.argsort(row)[::-1][:25]
    top_features=[(features[i], row[i]) for i in top_ids]
    df=pd.DataFrame(top_features,columns=['features','score'])
    return df

#The output of tfidf is a sparse matrix on which matrix operations donot work
#so a single row need to first be converted into dense format in order to apply top_tfidf_features function
def top_msg_features(X,features,row_id,top_words=25):
    row=np.squeeze(X[row_id].toarray())
    return top_tfidf_features(row,features,top_words)

#Now average tf-idf score has to be calculated of all emails(average per column of a tf-idf matrix)
def top_terms(X,features,grp_ids=None, min_tfidf=25):
    if grp_ids:
        msgs=X[grp_ids].toarray()
    else:
        msgs=X.toarray()
    msgs[msgs<min_tfidf]=0
#calculate the mean of each column across the selected rows which results in a single row of tf-idf values
    tfidf_means=np.mean(msgs,axis=0)
    return top_tfidf_features(tfidf_means,features,25)

#K-Means Clustering
#createclassifier with 3 clusters and 10 iterations
clf=KMeans(n_clusters=3, max_iter=100)
labels=clf.fit_predict(X)

#Now to visualise it we need to make 2d coordinates from the sparse matrix
X_dense=X.todense()
pca=PCA(n_components=2).fit(X_dense)
coords=pca.transform(X_dense)
label_colors=['r','b','g']
colors=[label_colors[i] for i in labels]
plt.scatter(coords[:,0],coords[:,1],c=colors)
plt.show()
    
#Now it is known which email the model assigned to each cluster,its possible to extract the top words per cluster
def top_words_per_cluster(X,y,features,min_tfidf=0.1,top_words=25):
    dfs=[]
    labels=np.unique(y)
    for label in labels:
        ids=np.where(y==label)
        features_df=top_terms(X,features,ids,min_tfidf= 0.1)
        features_df.label=label
        dfs.append(features_df)
    return dfs

def plot_top_words(dfs):
    fig=plt.figure(figsize=(11,8),facecolor="w")
    x=np.arange(len(dfs[0]))
    for i,df in enumerate(dfs):
        ax=fig.add_subplot(1,len(dfs),i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Tf-Idf Score",labelpad=16,fontsize=13)
        ax.set_title("cluster = "+str(df.label),fontsize=12)
        ax.ticklabel_format(axis='X',style='sci',scilimits=(-2,2))
        ax.barh(x,df.score,align='center',color='#7530FF')
        ax.set_yticks(x)
        ax.set_ylim([-1,x[-1]+1])
        yticks=ax.set_yticklabels(df.features)
    plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

tops=top_words_per_cluster(X,labels,features,0.1,25)
plot_top_words(tops)

        
