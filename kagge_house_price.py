
# coding: utf-8

# In[1]:


import pandas as pd
house_data_org=pd.read_csv(r"C:\Python27\Scripts\train.csv")
#print house_data_org.describe();


# In[2]:


#house_data_org.head()


# In[3]:


# Corecting the datatypes of coloumns of Data Frame. Some contains, NA NA which is not float but string
#house_data['Alley']=house_data['Alley'].replace({"NA": 'NotApplicable'})
#house_data.iloc[:,6]=str(house_data.iloc[:,6])
#house_data.iloc[:,57]=str(house_data.iloc[:,57])
#house_data.iloc[:,72]=str(house_data.iloc[:,72])
#house_data.iloc[:,73]=str(house_data.iloc[:,73])
#house_data.iloc[:,74]=str(house_data.iloc[:,74])
#print house_data.columns['Id']
dict_col={}
for i in range(0,len(house_data_org.iloc[0,:])):
    dict_col[i]=type(house_data_org.iloc[0,i])
#print dict_col
dict_dummy={}
print house_data_org.iloc[0,6]
for i in range(0,len(house_data_org.iloc[0,:])):
    if(dict_col[i]==str):
        dict_dummy[house_data_org.columns[i]] = list(house_data_org.iloc[:,i].unique())

#print dict_dummy
#print house_data.head()
#house_data['MasVnrType_BrkFace']=0
print house_data_org.head()


# In[22]:


house_data=house_data_org.copy()
for i in dict_dummy:
    arr=dict_dummy[i]
    for j in arr:
        print i,j
        house_data[str(i)+"_"+str(j)]="0"
    for k in range(0,len(house_data)):
        #house_data[str(i)+"_"+str(house_data[i][k])][k]=1
        house_data.loc[k,str(i)+"_"+str(house_data[i][k])]=1
    del house_data[i]
print house_data.head()
#print house_data.head()
#data=pd.DataFrame(dict_dummy['MasVnrType'])
#df=dict_dummy.get_dummies()


# In[20]:


#house_data['BldgType_Duplex']
#house_data['Id'].corr(house_data['MSSubClass'])
#house_data['Id'].corr(house_data['BldgType_Duplex'], method="spearman")

for i in range


# In[6]:


print house_data.describe()
import matplotlib.pyplot as plt
#plt.matshow(house_data.corr())
#plt.show()
#plt.hist will take the list as imput and will draw the histogram.
plt.hist(list(house_data.iloc[:,1]))
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
#fig=plt.gcf()
#import plotly as py
#plot_url = py.offline.plot(fig, filename='mpl-basic-histogram.html')
no_of_cols=len(house_data.iloc[0,:])
no_of_rows=len(house_data)
int_cols_dict=dict(house_data.mean())
print no_of_cols


# In[7]:



house_data.iloc[:,0].fillna(house_data.iloc[:,0].mean())


# In[8]:


#house_data.iloc[:,1].fillna(house_data.iloc[:,1].mean())
#type(house_data.iloc[0,7])
type(house_data.iloc[0,6])
#house_data.iloc[:,4].fillna(house_data.iloc[:,4].mean())


# In[9]:


#house_data['Id'].fillna(house_data['Id'].mean())
for i in range(0,7):
    if(not isinstance(house_data.iloc[0,i], str)):
        try:
            house_data.iloc[:,i].fillna(house_data.iloc[:,i].mean(), inplace=True)
        except:
            continue
house_data


# In[10]:


import sklearn
from sklearn.model_selection import train_test_split
house_data.corr()
x=house_data.iloc[:,0:-1]
y=house_data.iloc[:,-1]
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.33, random_state = 5)


# In[11]:


#from sklearn import linear_model
#lm = linear_model.LinearRegression()
#model = lm.fit(X_train, Y_train)
#predictions = lm.predict(X_test)

