#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#reading the dataset  using read.csv
data=pd.read_csv("C:/Users/manid/Downloads/spam.csv")
data
dataold=data.copy()
data=data.iloc[:,:2]


# In[3]:


data.columns=["labels","text"]
data


# In[4]:


#no of rows in dataset or datapoints
data['labels'].count()


# In[5]:


#no of columns in dataset
data.columns


# In[6]:


#no of class lables in dataset
data['labels'].value_counts()


# In[7]:


#plotting the histogram 
sns.histplot(data,x='labels')


# In[8]:


#ploting pie chart 
keys=["HAM","SPAM"]
d=[4825,747]
plt.pie(d,labels=keys,autopct='%.0f%%')
plt.show()


# In[9]:


#WordCloud is used to know the words which has higher frequency 
ham_words = ''.join(list(data[data["labels"]=='ham']['text']))
ham_wc = WordCloud(width =200,height = 200,background_color='lightblue').generate(ham_words)
plt.figure(figsize = (10, 8))
plt.imshow(ham_wc)
plt.axis('off')
plt.tight_layout(pad = 10)
plt.show()


# In[10]:


spam_words = ' '.join(list(data.loc[data['labels']=='spam','text']))
spam_wc = WordCloud(width = 200,height = 200,background_color='white').generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = None)
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 10)
plt.show()


# In[11]:


#removing stopwords from the dataset
#stopwords are the words that are commonly used in a language eg: is, not,and,the etc..
stopwords=set(stopwords.words("english"))


# In[12]:


x=data['text']
for i in range(0,len(x)):
    words=[word.lower() for word in x[i].split() if word.lower() not in stopwords]
    x[i]=' '.join(words)
print("Dataset after removing stopwords ")
data


# In[13]:


#removing special characters and punctuation marks from the dataset
x=data['text']
for i in range(len(x)):
    words=[word for word in x[i].split() if word.isalnum()]
    x[i]=' '.join(words)
print("Dataset after removing punctuations and special characters")
data


# In[14]:


#stemming is a technique of changing the words to root words eg : lives ,living,lived to live
x=data['text']
snow_stemmer=SnowballStemmer(language='english')
for i in range(len(x)):
    words=[snow_stemmer.stem(word) for word in x[i].split()]
    x[i]=' '.join(words)
print(data)


# In[15]:


#coverting text to vector form 
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(data['text'].values)
count=x.toarray()
count=pd.DataFrame(count,columns=vectorizer.get_feature_names())
finaldata=pd.concat([data['labels'],count],axis=1)
print(finaldata)


# In[16]:


#splitting dataset into train and test
xtrain,xtest,ytrain,ytest=train_test_split(finaldata.iloc[:,1:],finaldata.iloc[:,0],test_size=0.2)


# In[17]:


xtrain.shape


# In[18]:


xtest.shape


# In[19]:


#building model for Naive bayes--
model=MultinomialNB(alpha=0.5,class_prior=[4825/5572,747/5572])
model.fit(xtrain,ytrain)


# In[20]:


#predicting the testdata using the above model
pred=model.predict(xtest)


# In[21]:


#building confusion matrix 
cf_matrix=confusion_matrix(ytest,pred)


# In[22]:


sns.heatmap(cf_matrix, annot=True) 
plt.show()


# In[23]:


#accuracy of the model
model.score(xtest,ytest)


# In[24]:


#testing the model using the sample data 
sample=["hi how are you "]
x=vectorizer.transform(sample)
x=x.toarray()
x=pd.DataFrame(x)
model.predict(x)


# In[25]:


sample=["you have won cash prize 10000$. claim it now"]
x=vectorizer.transform(sample)
x=x.toarray()
x=pd.DataFrame(x)
model.predict(x)


# In[26]:


sample=["Send 10 ruppes to these number 8500829989 to get cash prize upto 10000 dollars"] 
x=vectorizer.transform(sample)
x=x.toarray()
x=pd.DataFrame(x)
model.predict(x)


# In[27]:


cf_matrix


# In[ ]:





# In[ ]:




