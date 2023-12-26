#!/usr/bin/env python
# coding: utf-8

# # Business Understanding 
# - Forecasting transaction 
# - Likley reg cuz cont prediction
# - Data for 3 years 
# - Data quality is okay

# # Data Understanding 

# In[146]:


import pandas as pd


# In[190]:


df=pd.read_csv('regression.csv')
df.head()


# In[191]:


df.tail()


# In[192]:


df.info()


# In[193]:


# Assuming df is your DataFrame
for col in df.columns:
    print(col,len(df[col].unique()),df[col].unique())


# In[194]:


df.dtypes


# In[195]:


df.describe()


# # Visualize Data 

# In[196]:


import matplotlib.pyplot as plt 
import seaborn as sns 


# In[197]:


plt.figure(figsize=(20,6)) #spread of our cols
sns.violinplot(x='Account Type',y='Amount',data=df).set_title('Account Type ViolinPlot');


# In[198]:


plt.figure(figsize=(20,6)) #spread of our cols'Revenue' 'Expense' 'Asset' 'Liability'
sns.violinplot(x='Account Type',y='Amount',data=df[df['Account Type']=='Liability']).set_title('Account Type ViolinPlot');


# In[199]:


plt.figure(figsize=(20,6)) #spread of our cols'Revenue' 'Expense' 'Asset' 'Liability'
sns.violinplot(x='Account',y='Amount',data=df[df['Account Type']=='Revenue']).set_title('Account Type ViolinPlot');


# In[200]:


monthmap={
    'Jan':1,
    'Feb':2,
    'Mar':3,
    'Apr':4,
    'May':5,
    'Jun':6,
    'Jul':7,
    'Aug':8,
    'Sep':9,
    'Oct':10,
    'Nov':11,
    'Dec':12  
}


# In[201]:


df['Period']=df['Month'].apply(lambda x:monthmap[x])
df


# In[202]:


df[df['Month']=='Dec'].head()


# In[203]:


df['Day'] =1


# In[204]:


df['Date']=df['Year'].astype(str)+'-'+df['Period'].astype(str)+'-'+df['Day'].astype(str)


# In[205]:


df


# In[206]:


df['Date'].dtype


# In[207]:


df['Date']=pd.to_datetime(df['Date'])


# In[208]:


df['Date'].dtype


# In[209]:


df.dtypes


# In[210]:


plt.figure(figsize=(20,6))
sns.lineplot(x='Date',y='Amount',hue='Account Description',estimator=None,data=df[df['Account Type']=='Revenue']);


# # Correlation

# In[211]:


df_c=df.drop(['Month','Cost Centre','Account Type','Account Description'],axis=1)
df_c.corr()


# create df contail each account and the amount of it 

# In[213]:


corrdict ={}
for key,row in df.join(pd.get_dummies(df['Account'])).iterrows():
    corrdict[key] = {int(row['Account']):row['Amount']}


# In[214]:


corrdf=pd.DataFrame.from_dict(corrdict)


# In[215]:


corrdf


# In[216]:


corrdf=pd.DataFrame.from_dict(corrdict).T.fillna(0)


# In[217]:


corrdf


# In[218]:


corrdf.corr()


# In[219]:


sns.heatmap(corrdf.corr());


# In[220]:


df[df['Account']==3000000] #strong correlation 


# In[221]:


df[df['Account']==4000001]


# # Data Preparation 

# In[222]:


df['Account'].unique()


# In[223]:


import numpy as np 


# In[224]:


for account in df['Account'].unique():
    plt.figure(figsize=(20,6))
    sns.lineplot(x='Date',y='Amount',estimator=np.median,hue='Account Description',data=df[df['Account']==account]).set_title('{} by month'.format(account))


# In[225]:


# Remove it 
df=df[df['Account']!=3000001] #not has the same موسمية 


# In[226]:


df['Account'].unique()


# convert fields to correct data type

# In[183]:


df.dtypes


# In[233]:


df['Account']='ACC' + df['Account'].astype(str)
df['Year']=df['Year'].astype('object')


# In[234]:


df.dtypes


# In[237]:


# will remove period ,day and date cols cuz we create them for analysis 
df.drop(columns=['Period','Day','Date'],inplace=True)


# In[239]:


df


# combine Account and account describtion on one col

# In[243]:


len(df['Account'].unique())


# In[244]:


len(df['Account Description'].unique())


# In[245]:


df['AccountVal']=df['Account'] + df['Account Description']


# In[246]:


df.head()


# In[247]:


df.drop(columns=['Account','Account Description'],inplace=True)


# In[249]:


df.head()


# One hot encoding 

# In[251]:


df=pd.get_dummies(df)


# In[254]:


df = df.astype(int)


# In[255]:


df.dtypes


# # Modelling

# In[257]:


x =df.drop('Amount',axis=1)
y =df['Amount']


# In[260]:


#to reduce variance
from sklearn.model_selection import train_test_split


# In[268]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[269]:


print(x_train.shape , x_test.shape , y_train.shape , y_test.shape)


# import Dependencies

# In[274]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso , ElasticNet
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor


# In[275]:


pipelines ={
    'rf':make_pipeline(RandomForestRegressor(random_state=1234)),
    'gb':make_pipeline(GradientBoostingRegressor(random_state=1234)),
    'ridge':make_pipeline(Ridge(random_state=1234)),
    'lasso':make_pipeline(Lasso(random_state=1234)),
    'enet':make_pipeline(ElasticNet(random_state=1234)),    
}


# In[300]:


#same keys of the pipline
hypergrid = {
    'rf': {
        'randomforestregressor__min_samples_split': [2, 4, 6],
        'randomforestregressor__min_samples_leaf': [1, 2, 3]
    },
    'gb': {
        'gradientboostingregressor__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    },
    'ridge': {
        'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    },
    'lasso': {
        'lasso__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    },
    'enet': {
        'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    }
}


# In[301]:


fit_models={}
for algo,pipeline in pipelines.items():
    print(pipeline)


# In[302]:


from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError


# In[303]:


fit_models={}
for algo,pipeline in pipelines.items():
    model = GridSearchCV(pipeline,hypergrid[algo],cv=10,n_jobs=-1)
    try:
        print('Starting training for {}'.format(algo))
        model.fit(x_train,y_train)
        fit_models[algo] = model
        print('{} has been successfully fit.'.format(algo))
    except NotFittedError as e:
        print(repr(e))


# In[304]:


#make predictions
fit_models['rf'].predict(x_test)


# In[305]:


fit_models['ridge'].predict(x_test)


# # Evaluation

# to know the model that is performing the best 

# In[309]:


from sklearn.metrics import r2_score,mean_absolute_error


# In[315]:


for algo,model in fit_models.items():
    yhat = model.predict(x_test)
    print('{} scores - R2:{} MAE:{}'.format(algo,r2_score(y_test,yhat),mean_absolute_error(y_test,yhat)))


# In[318]:


best_model = fit_models['rf']


# # Deployment
