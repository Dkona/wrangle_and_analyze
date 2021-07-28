#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Gathering the data

# In[16]:


#Extracting data from twitter archive
df=pd.read_csv('twitter-archive-enhanced.csv')
#Extracting data from tsv file
df2 = pd.read_csv("image-predictions.tsv",sep='\t')


# In[17]:


#Converting data from tweepy to a data frame
df3 = pd.DataFrame(columns=['tweet_id', 'retweet_account', 'favorite_count'])
with open('tweet_json.txt') as f:
    for line in f:
        status = json.loads(line)
        tweet_id = status['id_str']
        retweet_count = status['retweet_count']
        favorite_count = status['favorite_count']
        df3 = df3.append(pd.DataFrame([[tweet_id, retweet_count, favorite_count]],
                            columns=['tweet_id', 'retweet_count', 'favorite_count']))

df3 = df3.reset_index(drop=True)


# # Assessing Data

# In[18]:


df.info()


# In[19]:


df2.info()


# In[20]:


df3.info()


# In[21]:


df.head()


# In[22]:


df2.head()


# In[23]:


df3.head()


# # Cleaning Data

# Quality - 9 issues
# 
# -tweet_id should be changed from string to numeric data type, so we can merge on tweet_id
# 
# -Remove retweets column as not relevant to the project
# 
# -Define: drop in_reply_to_user_id, in_reply_to_status_id as all values are null, and retweet columns as we dont want to look at retweets
# 
# -extracting the numerator from the ratings. Convert rating_numerator to a float so we can compare to the new extracted numerator values to find discrepancies
# 
# - extracting the Denominator from the ratings, #convert str to int
# 
# - convert timestamp to datetime datatype
# 
# - Extract doggo, puppo, pupper, blep, snoot, floof and then change to lowercase for consistency
# 
# - drop the old doggo, puppo, pupper, blep, snoot, rating_numerator, rating_denominator columns as the names with the upper case were not captured
# 
# - extract the source name from source column and drop the old columns
# 
# Tidiness - 2 issues
# 
# - Merging all three data files into a combined file
# 
# --Forming a Dog column with name of the dog with the most probability and that it is also True
# 
# 

# #Define: Merging all three data files into a combined file. Tweet_id should be changed from string to numeric data type, so we can merge on tweet_id

# Code:

# In[24]:


df3['tweet_id']=pd.to_numeric(df3["tweet_id"])
df = pd.merge(df, df3, how="outer", on=["tweet_id"])
df = pd.merge(df, df2, how="outer", on=["tweet_id"])


# Test:

# In[25]:


df.head()


# Define: Remove retweets column as not relevant to the project

# Code:

# In[26]:


df= df.drop('retweet_account', axis=1)


# Test: 

# In[27]:


df.head()


# Define: drop in_reply_to_user_id, in_reply_to_status_id as all values are null, and retweet columns as we dont want to look at retweets

# Code

# In[28]:


df=df.drop('in_reply_to_user_id', axis=1)
df=df.drop('in_reply_to_status_id', axis=1)
df=df[~df.retweeted_status_id.notnull()]
df=df.drop(['retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_timestamp'], axis = 1)


# Test

# In[29]:


df.head()


# #Define: extracting the numerator from the ratings. Convert rating_numerator to a float so we can compare to the new extracted numerator values to find discrepancies
# 

# Code

# In[30]:


df['Numerator']=df.text.str.extract('(\d+[\.]?[\d+]*/)')
df['Numerator']=df['Numerator'].str[0:-1].astype(float)
df["rating_numerator"] = df["rating_numerator"].astype(float)


# Test: to find any discrepancies. Reveals all decimal numbers were wrongly extracted in rating_numerator
# 

# In[31]:


df[df.Numerator!=df.rating_numerator]


# Define: extracting the Denominator from the ratings, #convert str to int
# 

# Code:
# 

# In[32]:


df['Denominator']=df.text.str.extract('/([0-9]+)')
df["Denominator"] = pd.to_numeric(df["Denominator"])


# Test

# In[33]:


df[df.Denominator!=df.rating_denominator]


# Define: convert timestamp to datetime datatype
# 

# Code:

# In[34]:


pd.to_datetime(df.timestamp)


# Test:

# In[35]:


df.info()


# #Define: Extract doggo, puppo, pupper, blep, snoot, floof and then change to lowercase for consistency

# #Code:

# In[36]:


df['Doggo']=df.text.str.extract('(doggo)(?i)')


# In[37]:


df['Puppo']=df.text.str.extract('(puppo)(?i)')


# In[38]:


df['Pupper']=df.text.str.extract('(pupper)(?i)')


# In[39]:


df['Blep']=df.text.str.extract('(blep)(?i)')


# In[40]:


df['Snoot']=df.text.str.extract('(snoot)(?i)')


# In[41]:


df['Floof']=df.text.str.extract('(floof)(?i)')


# In[42]:


df['Doggo']=df.Doggo.str.lower()


# In[43]:


df['Puppo']=df.Puppo.str.lower()


# In[44]:


df['Pupper']=df.Pupper.str.lower()


# In[45]:


df['Blep']=df.Blep.str.lower()


# In[46]:


df['Snoot']=df.Snoot.str.lower()


# In[47]:


df['Floof']=df.Floof.str.lower()


# We will compare the new doggo,floofer,pupper,puppo columns with the one in the data to see which ones were not extracted. We will have to replace None and NaN with "", so we can compare all cells (NaN were not comparable to NaN). We see that data that did not match were the ones with capital letters in the word. We were able to extract those in the new columns. 

# In[48]:


df['doggo'].replace(to_replace="None", value="", inplace=True)


# In[49]:


df['floofer'].replace(to_replace="None", value="", inplace=True)


# In[50]:


df['pupper'].replace(to_replace="None", value="", inplace=True)


# In[51]:


df['puppo'].replace(to_replace="None", value="", inplace=True)


# In[52]:


df['Doggo'].replace(to_replace=np.NaN, value="", inplace=True)


# In[53]:


df['Floof'].replace(to_replace=np.NaN, value="", inplace=True)


# In[54]:


df['Pupper'].replace(to_replace=np.NaN, value="", inplace=True)


# In[55]:


df['Puppo'].replace(to_replace=np.NaN, value="", inplace=True)


# #Define: drop the old doggo, puppo, pupper, blep, snoot, rating_numerator, rating_denominator  columns
# 

# Code:

# In[56]:


df=df.drop(['doggo', 'floofer', 'pupper', 'puppo','rating_numerator', 'rating_denominator'], axis = 1)


# Test:

# In[57]:


df.info()


# #Define:extract the source name from source column and drop the old columns
# 

# Code:

# In[58]:


df[['s1','Source_name','s3']]=df.source.apply(lambda x: pd.Series(str(x).split(">")))
df.Source_name=df.Source_name.str[:-3]
df=df.drop(['s1','s3','source','expanded_urls'],axis=1)


# Test:

# In[59]:


df.head()


# #Define: Forming a Dog column with name of the dog with the most probability that it is also True
# 

# #Code: change the cell to NaN if the p1_dog, p2_dog, p3_dog is False
# 

# In[60]:


for i in df.p1.index:
    if df.p1_dog[i]==False:
        df.p1[i]=np.NaN
for i in df.p2.index:
    if df.p2_dog[i]==False:
        df.p2[i]=np.NaN   
for i in df.p3.index:
    if df.p3_dog[i]==False:
        df.p3[i]=np.NaN


# Test:

# In[61]:


df.head()


# Define: If the p1, p2, or p3 values are null then dog name will be one which is not null

# Code:

# In[62]:


df['Dog']=""


# In[63]:


for i in df.index:
    if df.p1[i]!=np.NaN:
        df.Dog[i]=df.p1[i]
    elif df.p2[i]!=np.NaN:
        df.Dog[i]=df.p2[i]
    elif df.p3[i]!=np.NaN:
        df.Dog[i]=df.p3[i]
    else:
        df.Dog[i]=np.NaN
    


# In[64]:


df.Dog=df.Dog.str.capitalize()


# Test:

# In[65]:


df.head()


# Define: Transfer all clean files to csv file twitter_archive_master

# #Code:
# 

# In[66]:


df.to_csv("twitter_archive_master.csv")


# # Question: Which are the sources of the most tweets?

# In[67]:


df.groupby('Source_name').tweet_id.count().sort_values(ascending=False)


# This shows that Twitter for iPhone was the source with most tweets

# In[68]:


df.groupby('Source_name').tweet_id.count().plot(kind="pie")
ax = plt.axes()
plt.ylabel("")
plt.title('Sources of Tweets')


# # Question: Which tweets had the max rating?

# In[69]:


df.nlargest(10, 'Numerator', keep='all')


# This shows the top 10 records with the most ratings

# # Question:Which listings have the 5 largest favorite counts?

# In[70]:


df.favorite_count=pd.to_numeric(df.favorite_count)
df.nlargest(5,'favorite_count')


# The above results show the Dogs with the 5 largest favorite counts

# In[71]:


df.favorite_count.sort_values(ascending=False).nlargest(5).plot(kind='bar')
plt.ylabel('favorite_count')
plt.title('Type of Dogs with top 5 favorite counts')
ax = plt.axes()
ax.set_xticks([0, 1, 2,3,4])
ax.set_xticklabels(['Labrador_retriever', 'Lakeland_terrier', 'Chihuahua', 'French_bulldog', 'Eskimo_dog'])


# # Question: Which dogs have the highest ratings?
# 

# In[72]:


df.groupby('Dog').Numerator.mean().nlargest(10)


# The above results show the type of dogs with the top 10 ratings

# In[73]:


df.groupby('Dog').Numerator.mean().nlargest(10).plot(kind='bar')
plt.ylabel('ratings')
plt.title('Type of Dogs with top 10 ratings')
 

