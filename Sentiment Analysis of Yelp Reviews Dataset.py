
# coding: utf-8

# # Sentiment Analysis of Yelp Reviews Dataset
# 
# ### Dataset Description:
# The Dataset contains 10,000 Yelp reviews from the customer. The following are the dataset attributes:
# 1. business_id (ID for different business reviews)
# 2. date (review posting date)
# 3. review_id (ID for each review)
# 4. stars (rating of business in 1-5 range)
# 5. text (text of each review)
# 6. type (type of text)
# 7. used_id (ID of the user)
# 8. cool/useful/funny (comments of reviews)
# 
# ### Purpose of analysis:
# The purpose of this analysis is to apply sentiment analysis technique on the Yelp review dataset from the customers and predict whether the review is negative or positive. To do this task, we will be performing three predictive Machine Learning algorithm to build model. In the conclusion, we will identifying which model is best for sentiment reviews.
# 
# *** This dataset has been taken from Kaggle. You can find the dataset in the following link:
#     https://www.kaggle.com/omkarsabnis/sentiment-analysis-on-the-yelp-reviews-dataset/data   ***
#     
# The following steps will be followed for the analysis:
# 1. Importing the Dataset
# 2. Processing the dataset
# 3. Vectorization
# 4. Splitting into Train and Test models
# 5. Applying classification method
# 6. Conclusion
# 

# ### Importing important libraries and uploading dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing NLTK library for using stop words method
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('E:/Mayank(D)/Northeastern University/quater 4/Python for Data science/yelp.csv')
df.head()


# In[3]:


print(df.shape)


# In[4]:


print(df.info())


# In[5]:


print(df.describe)


# In[6]:


#creating a new column for length of the text
df['text_len'] = df['text'].apply(len)
df.head()


# ### VIsualising the data

# In[7]:


graph1 = sns.FacetGrid(data=df, col='stars')
graph1.map(plt.hist, 'text_len', color = 'green', bins = 50)


# In[8]:


#Grouping the data using start rating and finding if any correlation 
stars = df.groupby('stars').mean()
stars.corr()
sns.heatmap(data = stars.corr(), annot = True)

#### This shows that funny is strongly correlated to useful and useful is strongly correlated to text_len
#### Thus, we can say that longer reviews are more funny and useful


# In[9]:


#Preparing for classification
df_class = df[(df['stars']==1) | (df['stars']==5)]
df_class.shape

#putting them in seperate variable
x = df_class['text']
y = df_class['stars']
print(x.head())
print(y.head())


# In[10]:


#Data cleaning by removing stop words and puntuation
import string
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# ### Performing Vectorization

# In[11]:


#Import countVectorizer and define it with a variable. Along with that we will fit it to our review text stored in x
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = text_process).fit(x)
print(len(vectorizer.vocabulary_))


# In[13]:


review_0 = x[0]
print(review_0)
vocab_0 = vectorizer.transform([review_0])
print(vocab_0)

print("Following Words back")
print(vectorizer.get_feature_names()[11443])
print(vectorizer.get_feature_names()[22077])


# In[19]:


#Now applying vectorization to the ful review set which would check the shape of new x
x = vectorizer.transform(x)
print('Shape of Sparse Matrix: ', x.shape)
print('Amount of Non-Zero occurrences: ', x.nnz)


# In[22]:


# Percentage of non-zero values
density = (100.0 * x.nnz / (x.shape[0] * x.shape[1]))
print("Density = ",density)


# ### Splitting the data into train and test

# In[24]:


#Splitting the dataset into training data and test data in the proportion of 80:20
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)


# ## Applying the classification method
# ### Multinomial Naive Bayes

# In[42]:


#Building the model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train, y_train)

#Testing our model
nb_predict = nb.predict(x_test)

#Creating the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, nb_predict))

print('\n')

#Creating the classification report
print(classification_report(y_test, nb_predict))  ### The model achieved 93% accuracy. 
                                                  ### However, since we know that there are some bias values, 
                                                  ### so let's just test it on a single review.

#positive single review
pos_review = df_class['text'][59]
pos_review

pos_review_t = vectorizer.transform([pos_review])
nb.predict(pos_review_t)[0]                       ### 5 star rating which is good as expected 


#Negative single review
neg_review = df_class['text'][281]
neg_review

neg_review_t = vectorizer.transform([neg_review])
nb.predict(neg_review_t)[0]                       ### 1 star rating which is fine as expected


# ### K-NN classifier

# In[48]:


#Building the model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

#Testing our model on x_test
knn_predict = knn.predict(x_test)

#Creating the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, knn_predict))

print('\n')

#Creating the classification report
print(classification_report(y_test, knn_predict))    ### The model achieved 78% accuracy


# ### Support Vector Machine

# In[54]:


#Building the model
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)

#Testing our model on x_test
svm_predict = svm.predict(x_test)

#Creating the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, svm_predict))

print('\n')

#Creating the classification report
print(classification_report(y_test, svm_predict))    ### The model achieved 67% accuracy


# ### Random Forest Classifier

# In[60]:


#Building the model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

#Testing our model on x_test
rf_predict = rf.predict(x_test)

#Creating the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, rf_predict))

print('\n')

#Creating the classification report
print(classification_report(y_test, rf_predict))    ### The model achieved 86% accuracy


# ## Conclusion 
# 
# #### After applying four machine learning algorithm to our model, we found different accuracy produced through eah of them: <br>
# 
# 1. MultiNomial Naive Bayes produced 93% accuracy
# 2. KNN produced 78% accuracy
# 3. SVM produced 67% accuracy
# 4. Random Forest produced 86% accuracy
# 
# #### By the above analysis, we can say that Naive Bayes and Random forest provides the best model.
# #### Among these two algorithms, since Naive Bayes gives the most accurate result, we have tested it on some single positive and single negative reviews which confirms that the model build is almost correct. But, there are some values which does not qualify the model should be because of bias data towards the positive reviews.
# 
