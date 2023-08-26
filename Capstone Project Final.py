#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_palette("deep")


# ### Business Objective
# 
# An E Commerce company or DTH (you can choose either of these two domains) provider is 
# facing a lot of competition in the current market and it has become a challenge to retain the 
# existing customers in the current situation. Hence, the company wants to develop a model 
# through which they can do churn prediction of the accounts and provide segmented offers to the 
# potential churners. In this company, account churn is a major thing because 1 account can have 
# multiple customers. hence by losing one account the company might be losing more than one customer.
# 
# You have been assigned to develop a churn prediction model for this company and provide 
# business recommendations on the campaign.
# 
# Your campaign suggestion should be unique and be very clear on the campaign offer because 
# your recommendation will go through the revenue assurance team. If they find that you are 
# giving a lot of free (or subsidized) stuff thereby making a loss to the company; they are not going 
# to approve your recommendation. Hence be very careful while providing campaign recommendation

# In[2]:


xls = pd.ExcelFile('D:/Capstone Project/Capstone Project/Customer+Churn+Data.xlsx')


# In[3]:


cust_churn = pd.read_excel(xls, 'Data for DSBA')
cust_churn.head(10)


# In[4]:


print(f'The number of rows in the dataset are {cust_churn.shape[0]}.')
print(f'The number of columns in the dataset are {cust_churn.shape[1]}.')


# In[5]:


cust_churn.info()


# We can observe from the above table that there are seven numerical attributes and twelve categorical attributes.

# In[6]:


cust_churn.isnull().sum()


# In[7]:


print(f'The total amount of nulls present in the dataset are {cust_churn.isnull().sum().sum()}.')


# In[8]:


cust_churn.Tenure.value_counts().tail(10)


# In[9]:


cust_churn['Tenure'] = cust_churn['Tenure'].replace('#', np.nan)


# In[10]:


cust_churn.Payment.value_counts()


# In[11]:


cust_churn.Gender.value_counts()


# In[12]:


cust_churn['Gender'] = cust_churn['Gender'].replace('M', 'Male')
cust_churn['Gender'] = cust_churn['Gender'].replace('F', 'Female')


# In[13]:


cust_churn.Gender.value_counts()


# In[14]:


cust_churn['Account_user_count'].value_counts()


# In[15]:


cust_churn['Account_user_count'] = cust_churn['Account_user_count'].replace('@', np.nan)


# In[16]:


cust_churn['account_segment'].value_counts()


# In[17]:


cust_churn['account_segment'] = cust_churn['account_segment'].replace('Regular +', 'Regular Plus')
cust_churn['account_segment'] = cust_churn['account_segment'].replace('Super +', 'Super Plus')


# In[18]:


cust_churn['account_segment'].value_counts()


# In[19]:


cust_churn['Marital_Status'].value_counts()


# In[20]:


cust_churn['rev_per_month'].value_counts().head(10)


# In[21]:


cust_churn['rev_per_month'] = cust_churn['rev_per_month'].replace('+', np.nan)


# In[22]:


cust_churn['rev_growth_yoy'].value_counts().tail()


# In[23]:


cust_churn['rev_growth_yoy'] = cust_churn['rev_growth_yoy'].replace('$', np.nan)


# In[24]:


cust_churn['coupon_used_for_payment'].value_counts().tail()


# In[25]:


cust_churn['coupon_used_for_payment'] = cust_churn['coupon_used_for_payment'].replace('#', np.nan)
cust_churn['coupon_used_for_payment'] = cust_churn['coupon_used_for_payment'].replace('$', np.nan)
cust_churn['coupon_used_for_payment'] = cust_churn['coupon_used_for_payment'].replace('*', np.nan)


# In[26]:


cust_churn['Day_Since_CC_connect'].value_counts().tail()


# In[27]:


cust_churn['Day_Since_CC_connect'] = cust_churn['Day_Since_CC_connect'].replace('$', np.nan)


# In[28]:


cnt = 0
for row in cust_churn['cashback']:
    try:
        float(row)
    except ValueError:
        cust_churn['cashback'].loc[cnt] = np.nan
    cnt += 1


# In[29]:


cust_churn['cashback'] = cust_churn['cashback'].astype('float64')


# In[30]:


cust_churn['Login_device'].value_counts()


# In[31]:


cust_churn['Login_device'] = cust_churn['Login_device'].replace('&&&&', np.nan)


# In[32]:


cust_churn['Login_device'].value_counts()


# In[33]:


cust_churn.info()


# In[34]:


print(f'The total amount of nulls present after cleaning the dataset are {cust_churn.isnull().sum().sum()}.')


# In[35]:


cust_churn.size


# In[36]:


print(f'The null values cover {round(((cust_churn.isnull().sum().sum())/(cust_churn.size))*100, 2)}% of the data.')


# In[37]:


plt.figure(figsize = (10,10))
sns.heatmap(cust_churn.isnull(), cbar = False, cmap = 'coolwarm', yticklabels = False)
plt.show()


# In[37]:


cust_churn.describe().T


# Inferences:
# 
# 

# In[38]:


print(f'The number of duplicated rows is {cust_churn.duplicated().sum()}.')


# Removing the 'AccountID' attribute as it is insignificant for use in the dataset.

# In[39]:


cust_churn.drop('AccountID', axis=1, inplace=True)


# ### Countplot of Categorical Variables

# In[40]:


cust_churn.columns


# In[41]:


cc_cat = cust_churn.select_dtypes(['object'])


# In[43]:


plt.figure(figsize=(12,8))
for i in range(len(cc_cat.columns)):
    plt.subplot(2,3,i+1)
    sns.countplot(data=cc_cat, x=cc_cat.columns[i])
    plt.grid(True)
    plt.xticks(rotation=60)
    plt.title(f'Countplot of {cc_cat.columns[i]}')
    plt.tight_layout()
    i+=1


# ### Imputing null values

# In[42]:


cust_churn['Payment'] = cust_churn['Payment'].fillna(cust_churn['Payment'].mode()[0])
cust_churn['Gender'] = cust_churn['Gender'].fillna(cust_churn['Gender'].mode()[0])
cust_churn['account_segment'] = cust_churn['account_segment'].fillna(cust_churn['account_segment'].mode()[0])
cust_churn['Marital_Status'] = cust_churn['Marital_Status'].fillna(cust_churn['Marital_Status'].mode()[0])
cust_churn['Login_device'] = cust_churn['Login_device'].fillna(cust_churn['Login_device'].mode()[0])


# In[43]:


for feature in cust_churn.columns:
    if cust_churn[feature].dtype == 'object':
        print('\n')
        print('feature:', feature)
        print(pd.Categorical(cust_churn[feature].unique()))
        print(pd.Categorical(cust_churn[feature].unique()).codes)
        cust_churn[feature] = pd.Categorical(cust_churn[feature]).codes


# In[46]:


cust_churn.dtypes


# In[44]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)


# In[45]:


cust_churn_imputed = pd.DataFrame(imputer.fit_transform(cust_churn), columns = cust_churn.columns)


# In[46]:


cust_churn_imputed.isnull().sum()


# ### Univariate Analysis

# In[45]:


cc_num = cust_churn_imputed[['Churn', 'Tenure', 'City_Tier', 'CC_Contacted_LY', 'Service_Score', 'Account_user_count',
       'CC_Agent_Score', 'rev_per_month', 'Complain_ly', 'rev_growth_yoy', 'coupon_used_for_payment', 
           'Day_Since_CC_connect', 'cashback']]


# In[51]:


plt.figure(figsize=(12,12))
for i in range(len(cc_num.columns)):
    plt.subplot(5,3,i+1)
    sns.histplot(data=cc_num, x=cc_num.columns[i], kde=True)
    plt.grid(True)
    plt.title(f'Histogram of {cc_num.columns[i]}')
    plt.tight_layout()
    i+=1


# In[52]:


# HNI - High Net worth Individuals
plt.figure(figsize=(12,12))
for i in range(len(cc_num.columns)):
    plt.subplot(5,3,i+1)
    sns.boxplot(data=cc_num, x=cc_num.columns[i])
    plt.grid(True)
    plt.title(f'Boxplot of {cc_num.columns[i]}')
    plt.tight_layout()
    i+=1


# ### Bivariate Analysis

# In[53]:


sns.pairplot(data=cc_num, diag_kind='kde');


# In[54]:


plt.figure(figsize=(12,12))
sns.set(font_scale=1)
sns.heatmap(cc_num.corr(), annot=True, fmt='.2f');
plt.title('Correlation Matrix', fontsize=20)
plt.show()


# (i) It seems that all of the variables have a low correlation with the target variable 'Churn'.
# 
# (ii) 'Day_Since_CC_connect' and 'Tenure' have a low inverse correlation with the target variable 'Churn' which means if value of one of these two variables increases then significantly the value of target variable 'Churn' decreases.
# 

# In[55]:


plt.figure(figsize=(6,6))
sns.boxplot(y=cust_churn_imputed.Tenure,x=cust_churn_imputed.Churn);
plt.title('Boxplot of Churn vs Tenure', fontsize=16)
plt.show()


# ### Outlier Treatment

# In[47]:


def remove_outlier(col):
    sorted(col)
    Q1, Q3 = np.percentile(col, [25,75])
    IQR = Q3-Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


# In[48]:


lrTenure,urTenure=remove_outlier(cust_churn_imputed['Tenure'])
cust_churn_imputed['Tenure']=np.where(cust_churn_imputed['Tenure']>urTenure,urTenure,cust_churn_imputed['Tenure'])


lrCC_Contacted_LY,urCC_Contacted_LY=remove_outlier(cust_churn_imputed['CC_Contacted_LY'])
cust_churn_imputed['CC_Contacted_LY']=np.where(cust_churn_imputed['CC_Contacted_LY']>urCC_Contacted_LY,urCC_Contacted_LY,cust_churn_imputed['CC_Contacted_LY'])


lrrev_per_month,urrev_per_month=remove_outlier(cust_churn_imputed['rev_per_month'])
cust_churn_imputed['rev_per_month']=np.where(cust_churn_imputed['rev_per_month']>urrev_per_month,urrev_per_month,cust_churn_imputed['rev_per_month'])


lrcoupon_used_for_payment,urcoupon_used_for_payment=remove_outlier(cust_churn_imputed['coupon_used_for_payment'])
cust_churn_imputed['coupon_used_for_payment']=np.where(cust_churn_imputed['coupon_used_for_payment']>urcoupon_used_for_payment,urcoupon_used_for_payment,cust_churn_imputed['coupon_used_for_payment'])


lrDay_Since_CC_connect,urDay_Since_CC_connect=remove_outlier(cust_churn_imputed['Day_Since_CC_connect'])
cust_churn_imputed['Day_Since_CC_connect']=np.where(cust_churn_imputed['Day_Since_CC_connect']>urDay_Since_CC_connect,urDay_Since_CC_connect,cust_churn_imputed['Day_Since_CC_connect'])


lrcashback,urcashback=remove_outlier(cust_churn_imputed['cashback'])
cust_churn_imputed['cashback']=np.where(cust_churn_imputed['cashback']>urcashback,urcashback,cust_churn_imputed['cashback'])
cust_churn_imputed['cashback']=np.where(cust_churn_imputed['cashback']<lrcashback,lrcashback,cust_churn_imputed['cashback'])


# In[49]:


selected_cols = cust_churn_imputed[['Tenure', 'CC_Contacted_LY', 'rev_per_month', 'coupon_used_for_payment', 
                    'Day_Since_CC_connect', 'cashback']]


# In[53]:


selected_cols


# In[50]:


plt.figure(figsize=(12,6))
for i in range(len(selected_cols.columns)):
    plt.subplot(2,3,i+1)
    sns.boxplot(data=selected_cols, x=selected_cols.columns[i])
    plt.grid(True)
    plt.title(f'Boxplot of {selected_cols.columns[i]}')
    plt.tight_layout()
    i+=1


# In[51]:


plt.figure(figsize=(12,12))
for i in range(len(cust_churn_imputed.columns)):
    plt.subplot(6,3,i+1)
    sns.boxplot(data=cust_churn_imputed, x=cust_churn_imputed.columns[i])
    plt.grid(True)
    plt.title(f'Boxplot of {cust_churn_imputed.columns[i]}')
    plt.tight_layout()
    i+=1


# In[51]:


cust_churn_imputed['Churn'].value_counts(normalize=True)


# ### KNN Elbow Curve

# In[52]:


from scipy.stats import zscore
from sklearn.cluster import KMeans


# In[53]:


cc_imputed_scaled = cust_churn_imputed


# In[54]:


wss_value = []

for i in range(1,18):
    km = KMeans(random_state=0, n_clusters=i)
    km.fit(cc_imputed_scaled)
    wss_value.append(km.inertia_)
    i+=1


# In[55]:


plt.figure(figsize=(7,7))
plt.plot(range(1,18), wss_value)
plt.xlabel('Number of Clusters')
plt.ylabel('WSS Values')
plt.title('Elbow Method')
plt.show()


# Number of clusters to be chosen from the above curve is equal to 3 i.e. k=3

# In[64]:


k_means = KMeans(random_state=1, n_clusters=3)
k_means.fit(cc_imputed_scaled)


# In[65]:


label = k_means.labels_


# In[66]:


cc_imputed_scaled['Labels'] = label


# In[67]:


cc_imputed_scaled[cc_imputed_scaled['Labels']==0].describe()


# In[68]:


cc_imputed_scaled[cc_imputed_scaled['Labels']==1].describe()


# In[69]:


cc_imputed_scaled[cc_imputed_scaled['Labels']==2].describe()


# In[71]:


from sklearn.metrics import silhouette_score
print(f'Hence the silhouette score is {round(silhouette_score(cc_imputed_scaled,label),4)}.')


# ### 1. Model building and interpretation   
# 
# #### a. Build various models (You can choose to build models for either or all of descriptive, predictive or prescriptive purposes) 
# 
# #### b. Test your predictive model against the test set using various appropriate performance metrics 
# (Classification report, confusion matrix, auc-roc curve - for classification data)
# 
# #### c. Interpretation of the model(s)
# 
# ### 2. Model Tuning
# 
# #### a.Ensemble modelling, wherever applicable 
# (Multiple models are created to predict an outcome by using many different modelling algorithms)
# 
# #### b. Any other model tuning measures(if applicable) 
# (Model tuning - Hyper parameter tuning)
# 
# #### c. Interpretation of the most optimum model and its implication on the business
# 
# 80-20 percent ratio split is used for the data and a random state (=1) is introduced in train test split.
# A 80/20 split is done to ensure a good accuracy of the model.

# In[46]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import tree


# In[47]:


X = cust_churn_imputed.drop('Churn', axis=1)
y = cust_churn_imputed.pop('Churn')


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


# ## CART Model

# In[109]:


param_grid1 = {
    'max_depth': [8,9],
    'min_samples_leaf': [10,15],
    'min_samples_split': [30,45],
    'max_features': [9]
}


# max depth- It is the number of nodes along the longest path from the root node down to the farthest leaf node. 
# 
# minimum sample leaf- It is the minimum number of samples required to be at a leaf node. It is usually selected at 1-3 percent of all records
# 
# minimum sample split- It is the minimum number of samples required to split an internal node. It is chosen to be usually three times of the chosen value of minimum sample leaf.
# 
# max features- It is usually a restriction given on high number of features to improve accuracy of the model. It is a hyper parameter which is usually chosen around half the number of available features.

# In[110]:


dt_model = DecisionTreeClassifier()


# In[111]:


grid_search1 = GridSearchCV(estimator=dt_model, param_grid=param_grid1, cv=3)


# 'cv' is nothing but an internal cross validation technique which is used to calculate the score for each combination of parameters in the grid. Here in this case cv is equal to 3.

# In[112]:


grid_search1.fit(X_train, y_train)


# In[113]:


grid_search1.best_params_


# In[114]:


best_grid1 = grid_search1.best_estimator_


# In[115]:


ytrain_predict = best_grid1.predict(X_train)
ytest_predict = best_grid1.predict(X_test)


# In[116]:


print(f'The accuracy score of the CART model (Train) is {round(grid_search1.score(X_train, y_train),5)}')


# In[117]:


print(classification_report(y_train, ytrain_predict))


# The bigger and most important problem that will be faced by the company is, when the model predicts that the customer is not at the verge of churning but actually he is at the verge of churning. That's when we are going to lose a precious customer. Such errors are called Type-2 error or also called as False Negatives. Such errors are determined by the recall score of the model. Hence recall scores have a high significance for this dataset. 

# In[65]:


from sklearn import metrics


# In[119]:


sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[120]:


ytrain_predict1 = best_grid1.predict_proba(X_train)
ytest_predict1 = best_grid1.predict_proba(X_test)


# In[121]:


probs1 = ytrain_predict1[:,1]
probs2 = ytest_predict1[:,1]


# In[122]:


auc1 = roc_auc_score(y_train, probs1)
print(f'The roc-auc score for the CART Model (Train) is {round(auc1,5)}')


# In[123]:


fpr, tpr, thresholds = roc_curve(y_train, probs1)

plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr, tpr, marker = '.')
plt.title('ROC Curve for Train Model', fontsize=16)
plt.show()


# In[124]:


print(f'The accuracy score of the CART model (Test) is {round(grid_search1.score(X_test, y_test),5)}')


# In[125]:


print(classification_report(y_test, ytest_predict))


# In[126]:


sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[127]:


auc2 = roc_auc_score(y_test, probs2)
print(f'The roc-auc score for the CART Model (Test) is {round(auc2,5)}')


# In[128]:


fpr, tpr, thresholds = roc_curve(y_test, probs2)

plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr, tpr, marker = '.')
plt.title('ROC Curve for Test CART Model', fontsize=16)
plt.show()


# ## Random Forest Model

# In[158]:


param_grid2 = {
    'max_depth': [9],
    'min_samples_leaf': [10],
    'min_samples_split': [45],
    'max_features':[9],
    'n_estimators':[201,251]
}


# n estimators- It is usually the number of trees we want to build. Higher the number of trees, better the performance but slower the code. 
# 
# As from the above decision tree model, we found the best parameters. Let's choose the hyper parameter for n estimators at random and find the best value.

# In[159]:


rfcl = RandomForestClassifier()
grid_search2 = GridSearchCV(estimator=rfcl, param_grid=param_grid2, cv=3)


# In[160]:


grid_search2.fit(X_train,y_train)


# In[161]:


grid_search2.best_params_


# In[162]:


best_grid2 = grid_search2.best_estimator_


# In[163]:


ytrain_predict2 = best_grid2.predict(X_train)
ytest_predict2 = best_grid2.predict(X_test)


# In[164]:


print(f'The accuracy score of the Random Forest Model (Train) is {round(grid_search2.score(X_train, y_train),5)}')


# In[165]:


print(classification_report(y_train, ytrain_predict2))


# In[166]:


sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict2)),annot=True,fmt='.5g',cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[167]:


ytrain_predict3 = best_grid2.predict_proba(X_train)
ytest_predict3 = best_grid2.predict_proba(X_test)

probs3 = ytrain_predict3[:,1]
probs4 = ytest_predict3[:,1]


# In[168]:


auc3 = roc_auc_score(y_train, probs3)
print(f'The roc-auc score for the Random Forest Model (Train) is {round(auc3,5)}')


# In[169]:


fpr, tpr, thresholds = roc_curve(y_train, probs3)

plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr, tpr, marker = '.')
plt.title('ROC Curve for Train Random Forest Model', fontsize=16)
plt.show()


# In[170]:


print(f'The accuarcy score of the Random Forest Model (Test) is {round(grid_search2.score(X_test, y_test),5)}')


# In[171]:


print(classification_report(y_test, ytest_predict2))


# In[172]:


sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict2)),annot=True,fmt='.5g',cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[173]:


auc4 = roc_auc_score(y_test, probs4)
print(f'The roc-auc score for the Random Forest Model (Test) is {round(auc4,5)}')


# In[174]:


fpr, tpr, thresholds = roc_curve(y_test, probs4)

plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr, tpr, marker = '.')
plt.title('ROC Curve for Test Random Forest Model', fontsize=16)
plt.show()


# ## ANN Model

# In[175]:


from sklearn.neural_network import MLPClassifier
# MLP - Multi Layered Perceptron


# In[176]:


param_grid3 = {
    'hidden_layer_sizes': [100, 150],
    'activation': ['logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'tol': [0.001],
    'max_iter': [10000]
}


# Hidden layer sizes are chosen on the basis of getting a better model. Sizes are allowed to be chosen as either 200 or 300.
# 
# Both the activation formulas are chosen i.e. logistic and relu (Rectified Linear Units)
# 
# For solver sgd (Stochastic Gradient Descent) and adam methods are chosen to get an optimal model.
# 
# Higher the tolerance value of the model, the faster the speed of the model but lower the accuracy. Lower the tolerance value of the model, higher the accuracy but slower the speed of the model. Hence, for this problem tolerance value is 0.001
# 
# For the model to reach optimal point has to run multiple iterations. So, the selected number of iterations is 10000.

# In[177]:


MLP = MLPClassifier()
grid_search3 = GridSearchCV(estimator=MLP, param_grid=param_grid3, cv=3)


# In[178]:


grid_search3.fit(X_train, y_train)


# In[179]:


grid_search3.best_params_


# In[180]:


best_grid3 = grid_search3.best_estimator_


# In[181]:


ytrain_predict4 = best_grid3.predict(X_train)
ytest_predict4 = best_grid3.predict(X_test)


# In[182]:


print(f'The accuracy score of the ANN Model (Train) is {round(grid_search3.score(X_train, y_train),5)}')


# In[183]:


print(classification_report(y_train, ytrain_predict4))


# In[184]:


sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict4)),annot=True,fmt='.5g',cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[185]:


ytrain_predict5 = best_grid3.predict_proba(X_train)
ytest_predict5 = best_grid3.predict_proba(X_test)

probs5 = ytrain_predict5[:,1]
probs6 = ytest_predict5[:,1]


# In[186]:


auc5 = roc_auc_score(y_train, probs5)
print(f'The roc-auc score for the ANN Model (Train) is {round(auc5,5)}')


# In[187]:


fpr, tpr, thresholds = roc_curve(y_train, probs5)

plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr, tpr, marker = '.')
plt.title('ROC Curve for Train ANN Model', fontsize=16)
plt.show()


# In[188]:


print(f'The accuracy score of the ANN Model (Test) is {round(grid_search3.score(X_test, y_test),5)}')


# In[189]:


print(classification_report(y_test, ytest_predict4))


# In[190]:


sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict4)),annot=True,fmt='.5g',cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[191]:


auc6 = roc_auc_score(y_test, probs6)
print(f'The roc-auc score for the ANN Model (Test) is {round(auc6,5)}')


# In[192]:


fpr, tpr, thresholds = roc_curve(y_test, probs6)

plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr, tpr, marker = '.')
plt.title('ROC Curve for Test ANN Model', fontsize=16)
plt.show()


# ## Backward Elimination Feature

# In[193]:


import statsmodels.formula.api as SM


# Statsmodel requires the labelled data, therefore, concatinating the y label to the train set.

# In[194]:


Default_train = pd.concat([X_train,y_train], axis=1)
Default_test = pd.concat([X_test,y_test], axis=1)


# In[195]:


Default_train["Churn"].value_counts(normalize=True)


# In[196]:


Default_train.columns


# In[197]:


#Creating our first model using all variables
model_1 = SM.logit(formula = 'Churn ~ Tenure + City_Tier + CC_Contacted_LY + Payment + Gender + Service_Score + Account_user_count + account_segment + CC_Agent_Score + Marital_Status + rev_per_month + Complain_ly + rev_growth_yoy + coupon_used_for_payment + Day_Since_CC_connect + cashback + Login_device', data=Default_train).fit()


# In[198]:


model_1.summary()


# In[199]:


# Removing Service_score in the next model as it has a probability value greater than 0.05

model_2 = SM.logit(formula = 'Churn ~ Tenure + City_Tier + CC_Contacted_LY + Payment + Gender + Account_user_count + account_segment + CC_Agent_Score + Marital_Status + rev_per_month + Complain_ly + rev_growth_yoy + coupon_used_for_payment + Day_Since_CC_connect + cashback + Login_device', data=Default_train).fit()


# In[200]:


model_2.summary()


# In[201]:


# Removing payment in the next model as it has a probability value greater than 0.05

model_3 = SM.logit(formula = 'Churn ~ Tenure + City_Tier + CC_Contacted_LY  + Gender + Account_user_count + account_segment + CC_Agent_Score + Marital_Status + rev_per_month + Complain_ly + rev_growth_yoy + coupon_used_for_payment + Day_Since_CC_connect + cashback + Login_device', data=Default_train).fit()


# In[202]:


model_3.summary()


# In[203]:


y_prob_pred_train = model_3.predict(Default_train)


# In[204]:


y_class_pred=[]
for i in range(0,len(y_prob_pred_train)):
    if np.array(y_prob_pred_train)[i]>0.5:
        a=1
    else:
        a=0
    y_class_pred.append(a)


# In[205]:


sns.heatmap((metrics.confusion_matrix(Default_train['Churn'],y_class_pred)),annot=True,fmt='.5g',cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[206]:


print(metrics.classification_report(Default_train['Churn'],y_class_pred))


# In[207]:


y_prob_pred_test = model_3.predict(Default_test)


# In[208]:


y_class_pred=[]
for i in range(0,len(y_prob_pred_test)):
    if np.array(y_prob_pred_test)[i]>0.5:
        a=1
    else:
        a=0
    y_class_pred.append(a)


# In[209]:


sns.heatmap((metrics.confusion_matrix(Default_test['Churn'],y_class_pred)),annot=True,fmt='.5g',cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[210]:


print(metrics.classification_report(Default_test['Churn'],y_class_pred))


# Above model is an under fitting model. It has a recall score of 0.48 in the test data which is greater than the recall score (= 0.47) of train data which shows that the model is under fitting.

# ## SMOTE using Logistic Regression

# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE 


# In[50]:


LogR = LogisticRegression(max_iter=10000)


# In[51]:


selector = RFE(estimator = LogR, n_features_to_select = 10)


# In[52]:


selector = selector.fit(X_train, y_train)


# In[53]:


selector.n_features_


# In[54]:


selector.ranking_


# In[55]:


sm = SMOTE(random_state=33, sampling_strategy = .75)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[56]:


selector_smote = selector.fit(X_res, y_res)


# In[57]:


selector_smote.n_features_


# In[58]:


pred_train_smote = selector_smote.predict(X_res)
pred_test_smote = selector_smote.predict(X_test)


# In[59]:


prob_train_smote = selector_smote.predict_proba(X_res)


# In[60]:


prob_test_smote = selector_smote.predict_proba(X_test)


# In[61]:


probs7 = prob_train_smote[:,1]
probs8 = prob_test_smote[:,1]


# In[62]:


print(f'The accuracy score of the SMOTE Model (Train) is {round(selector_smote.score(X_res, y_res),5)}')


# In[63]:


print(classification_report(y_res, pred_train_smote))


# In[66]:


sns.heatmap((metrics.confusion_matrix(y_res, pred_train_smote)),annot=True,fmt='.5g',cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[67]:


auc7 = roc_auc_score(y_res, probs7)
print(f'The roc-auc score for the SMOTE Model (Train) is {round(auc7,5)}')


# In[68]:


fpr, tpr, thresholds = roc_curve(y_res, probs7)

plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr, tpr, marker = '.')
plt.title('ROC Curve for Train SMOTE Model', fontsize=16)
plt.show()


# In[69]:


print(f'The accuracy score of the SMOTE Model (Test) is {round(selector_smote.score(X_test, y_test),5)}')


# In[70]:


print(classification_report(y_test, pred_test_smote))


# In[71]:


sns.heatmap((metrics.confusion_matrix(y_test, pred_test_smote)),annot=True,fmt='.5g',cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);


# In[72]:


auc8 = roc_auc_score(y_test, probs8)
print(f'The roc-auc score for the SMOTE Model (Train) is {round(auc8,5)}')


# In[73]:


fpr, tpr, thresholds = roc_curve(y_test, probs8)

plt.plot([0,1],[0,1], linestyle = '--')
plt.plot(fpr, tpr, marker = '.')
plt.title('ROC Curve for Test SMOTE Model', fontsize=16)
plt.show()


# In[7]:


train_accuracy = {'CART':0.9191, 'RF':0.9312, 'ANN':0.9245, 'Logit':0.8900, 'SMOTE':0.8117}
test_accuracy = {'CART':0.9045, 'RF':0.9210, 'ANN':0.9232, 'Logit':0.8900, 'SMOTE':0.8188}

train_recall = {'CART':0.72, 'RF':0.68, 'ANN':0.82, 'Logit':0.47, 'SMOTE':0.78}
test_recall = {'CART':0.68, 'RF':0.64, 'ANN':0.81, 'Logit':0.48, 'SMOTE':0.77}

train_F1 = {'CART':0.75, 'RF':0.77, 'ANN':0.79, 'Logit':0.58, 'SMOTE':0.78}
test_F1 = {'CART':0.70, 'RF':0.73, 'ANN':0.78, 'Logit':0.59, 'SMOTE':0.58}

train_roc_score = {'CART':0.9501, 'RF':0.973, 'ANN':0.9627, 'SMOTE':0.8734}
test_roc_score = {'CART':0.9295, 'RF':0.9576, 'ANN':0.9488, 'SMOTE':0.7413}

train_precision = {'CART':0.79, 'RF':0.88, 'ANN':0.75, 'Logit':0.77, 'SMOTE':0.78}
test_precision = {'CART':0.72, 'RF':0.84, 'ANN':0.75, 'Logit':0.77, 'SMOTE':0.47}


# In[8]:


comparison_table = pd.DataFrame({'Accuracy(Train)':train_accuracy, 'Accuracy(Test)':test_accuracy,
                                 'Recall(Train)':train_recall, 'Recall(Test)':test_recall,
                                 'F1 score(Train)':train_F1, 'F1 score(Test)':test_F1,
                                 'Precision(Train)':train_precision, 'Precision(Test)':test_precision,
                                 'roc-auc score(Train)':train_roc_score, 'roc-auc score(Test)':test_roc_score})


# In[9]:


comparison_table.T

