#!/usr/bin/env python
# coding: utf-8

# In[258]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import math
sns.set()

import warnings
warnings.filterwarnings("ignore")


# In[259]:


train = pd.read_csv("titanic data/train.csv")
test = pd.read_csv("titanic data/test.csv")


# In[260]:


train.head()


# In[261]:


train.shape

Describe  the train data set
# In[262]:


train.describe()


# In[263]:


#Describe include = ['0'] *will show the descriptive statics of object data types.
train.describe(include=['O']) 


# In[264]:


train.info()


# In[265]:


#Checking if any column has some missing values
train.isnull().sum()


# Note:
# -There are 177 rows with missing age, 687 rows with missing
# cabin and 2 rows with missing Embarked information
Looking into the testing dataset
# In[266]:


test.shape


# Note:
# -Test data has 418 rows and 1 columns.
#   Train data rows= 891, Test data rows=418,
#     Total rows = 891+418= 1309
# -We can see that 2/3 of total data is set as train and 
# around 1/3 of total data is set as Test data.

# In[267]:


test.head()


# NOte:
# -Survived coulumn is not present in the TEst data.
# -We have to train our classifier using the train and generate
# predictions(Survuved) on Test Data.

# In[268]:


test.info()


# NOte:
# -There are missing entries for Age in Test dataset as well.
# -OUt of 418 rows in test dataset, only 332 rows have age value.
# -Cabin values also missing in many rows. Only 91 rows have data out of total 418.

# In[269]:


test.isnull().sum()


# NOte:
# - There are 86 rows with missing age, 327 rows with missing cabin and 1 row wih missing Fare information. 
Relationship between Features and Survival:

-In this section, we analyze relationship between features with respect to Survival. We see how different featires values show different survival chance. We also plot different kinds of diagrams to visualize our data and findings.
# In[270]:


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print(len(survived))
print(len(not_survived))

print("Survived : {0} ({1}%)".format(len(survived),round((len(survived))/len(train) * 100.0, 2)))
print("Not Survived : {0} ({1}%)".format(len(not_survived), round((len(not_survived)/len(train) * 100), 2)))
print("Survived : %i (%.1f%%)"%(len(survived), float(len(survived))/len(train) * 100.0))
print("Not Survived : %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print("Total: %i" %len(train))

Finding relations between features and survival:
    1)Pclass vs Survival
# In[271]:


print(train.Pclass.value_counts())
print("Total:", len(train))


# In[272]:


pclass_survived = train.groupby('Pclass').Survived.value_counts()
pclass_survived


# In[273]:


#Plotting the pclass vs survived

pclass_survived.unstack(level=0).plot(kind='bar', subplots= False)


# Note:
#     -Higher class passangers have better survival chance(may be because they are more privilege to be saved).

# In[274]:


pclass_survived_average = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean()
pclass_survived_average


# In[275]:


pclass_survived_average.unstack(level=1).plot(kind = 'bar', subplots=False)
pclass_survived_average.plot(kind='bar', subplots=False)


# NOTE:
# -Higher class passengers (low Pclass) have better avarage survival than the low class(high Pclass) passengers.

# In[276]:


# The above statement can be clearly understood from the plot below.

sns.barplot(x='Pclass', y='Survived', data=train)

2) Sex vs. Survival
# In[277]:


train.Sex.value_counts()


# In[278]:


sex_survival = train.groupby('Sex').Survived.value_counts()
sex_survival


# In[279]:


male_not_survive = train[(train['Survived']==0) & (train['Sex']=='male')]
len(male_not_survive)


# In[280]:


sex_survival.unstack(level=0).plot(kind='bar', subplots = False)


# In[281]:


sex_survived_average = train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
sex_survived_average


# In[282]:


sex_survived_average.plot(kind='bar', subplots=False)


# In[283]:


sns.barplot(x='Sex', y='Survived', data=train)


# Note:
# -Females have better survival chance.
3) Pclass & Sex vs Survival 

-Below, I have just found out how many males and females are there in each Pclass then plotted a bar diagram with that infomration and found that there are more males among the 3rd Pclass passangers.
# In[284]:


tab = pd.crosstab(train['Pclass'], train['Sex'])
print(tab)

tab.div(tab.sum(1).astype(float), axis=0).plot(kind='bar', stacked= False)

plt.xlabel('Pclass')
plt.ylabel('Percentage')


# In[285]:


sns.factorplot('Sex', 'Survived', hue = 'Pclass', size= 4, aspect=2, data=train)


# NOTE:
# -From the above plot, it can be seen that:
#     -Women from 1st and 2nd Pclass have almost 100% survival chance.
#     -Men from 2nd and 3rd Pclass haev only around 10% survival chance:
4) Pclass, Sex & embarked vs Survival 
# In[286]:


sns.factorplot(x='Pclass', y= 'Survived', hue='Sex', col='Embarked', data=train)


# NOTE:
# From the above plot, it can be seen that:
#     -Almost all females from Pclss 1 and 2 survived.
#     -Females dying were mostly from 3rd Pclass.
#     -Males from Pclass 1 only have slighter higher survival chance than Pclass 2 and 3.
#     
5) Embarked vs Survived
# In[287]:


train.Embarked.value_counts()


# In[288]:


train.groupby('Embarked').Survived.value_counts()


# In[289]:


embarked_survived_average= train[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()
embarked_survived_average


# In[290]:


#train.groupby('Embarked').Survived.mean().plot(kind='bar')
sns.barplot(x='Embarked', y='Survived', data = train)

6) Parch vs Survival
# In[291]:


train.Parch.value_counts()


# In[292]:


train.groupby('Parch').Survived.value_counts()


# In[293]:


train[['Parch', "Survived"]].groupby(['Parch'], as_index=False).mean()


# In[294]:


#train.groupby('Parch').Survived.mean().plot(kind='bar')

sns.barplot(x='Parch', y='Survived', ci= None, data=train)
#ci= None will  hide the error bar

7)SibSp vs. Survival
# In[295]:


train.SibSp.value_counts()


# In[296]:


train.groupby('SibSp').Survived.value_counts()


# In[297]:


train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()


# In[298]:


#train.group by('SibSp').Survived.mean().plot(kind='bar')
sns.barplot(x='SibSp', y='Survived', ci= None, data=train)

#Ci= None will hide the error bar

8) Age vs. Survival
# In[299]:


fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split= True, ax=ax1)
sns.violinplot(x="Pclass", y= "Age", hue= "Survived", data=train, split=True, ax=ax2)
sns.violinplot(x="Sex", y='Age', hue= "Survived", data= train, split=True, ax=ax3)


# NOTE:
# 1) From Pclass violinplot, we can see that:
#     -1st Pclass has very few children as compared to other two clases.
#     -1st Pclass has more old people as compared to other two classes.
#     -Almost all children(between age 0 to 10) of 2nd Pclass survived.
#     -Most children of 3rd Pclass  survived.
#     -Younger people of 1st Pclass survived as compared to its older people.
#     

# 2) From Sex violinplot, we can see that:
#     -MOst male children (between age 0to 14) survived.
#     -Females with age between 18 to 40 have better survival chance.
plotting some distribution plots based on survival's sex
# In[300]:


total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]

male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]
female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]

male_not_survived = train[(train['Survived']==0) & (train['Sex']=='male')]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=='female')]


# In[301]:


len(male_not_survived)
len(female_not_survived)


# In[302]:


plt.figure(figsize=[15,5])
plt.subplot(111)
print(sns.distplot(total_survived['Age'].dropna().values, bins=range(0,81,1), kde= True, color = 'blue'))
print(sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0,81,1), kde=True, color='red', axlabel='Age'))


# In[303]:


plt.figure(figsize=[15,5])

plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0,81,1), kde=True, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0,81,1), kde= True, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0,81,1), kde= True, color= 'blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0,81,1), kde= True, color= 'red', axlabel='Male Age')


# NOTE:
# From the above figures, we can see that:
#     -Combining both male and female, we can see that children with age between 0 to 5 have better chance of survival.
#     -Females with age between "18 to 40" and 50 and above have higher chance of survival.
#     -Males with age between 0 to 14 have better chance of survival.
Correlating Features
# In[304]:


plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId', axis=1).corr(),vmax=0.6, square=True, annot= True)


# Note: 
# Heatmap of Correlation between differnet features:
#     -Positive numbers = Positive correlation, i.e. increase in one feature will increase the other feature & vice-versa.
#     -Negetive numbers = Negetive correlation, i.e. increase in one feature will decrease the other feature & vice-versa.
# In our case, we focus on which features have strong positive or negetive correlation with the Survived feature.
Feature Extraction:
    In this section, we select the appropriate features to train our classifier.
    Here, we create new features based on existing features.
    We also convert categorical features into numeric form.1) Name Feature:
    -Let's first extract titles from Name column.
# In[305]:


#combining train and test dataset
train_test_data = [train, test]

#extract titles from Nmae column.
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.')


# In[306]:


train.head()


# As you can see above, we have added a new column named Title in the Train dataset with the Title present in the particular passenger name.

# In[307]:


pd.crosstab(train['Title'], train['Sex'])


# NOTE:
# The number of passengers with each Title is shown above.
# We now replace some less common titles with the name "Other".

# In[308]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',  	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[309]:


sns.barplot(x='Title', y='Survived', ci=None, data=train)

Now, we convert the categorical Title values into numeric form.
# In[310]:


title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[311]:


train.head()

2) Sex Feature
# We convert the categorical value of Sex into numeric. We represent 0 as female and 1 as male.

# In[312]:


for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[313]:


train.head()

3) Embarked Feature
# There are empty values for some rows for Embarked column.
# The empty values are represented as "nan" in below list

# In[314]:


train.Embarked.value_counts()


# Note:
#     -> We find that category "S" has maximum passengers.
#     -> Hence, we replace 'nan' values with "S".

# In[315]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[316]:


train.head()


# In[317]:


train.Embarked.value_counts()


# We now convert the categorical value of Embarked into numeric.
# We represent 0 as S,1 as C and 2 as Q

# In[318]:


for dataset in train_test_data:
    #Print[dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map(
    {'S':0, 'C': 1, 'Q':2}).astype(int)


# In[319]:


train.head()

4) Age Feature
# We first will fill the NULL values of Age with a random number between (mean_age-std_age) and (mean_age+std_age).
# We then create a new column named AgeBand. This categorizes age into 5 different age range.

# In[320]:


for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['AgeBand'] = pd.cut(train['Age'], 5)

print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())


# In[321]:


train.head()


# In[322]:


train.head()


# Now, We map Age according to AgeBAnd

# In[323]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


# In[324]:


train.head()

5) Fare Feature
# Replace missing Fare values with the median of Fare.

# In[325]:


for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())


# Now, Create FareBand. We divide the Fare into 4 category range.

# In[326]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
print(train[['FareBand', 'Survived']].groupby(['FareBand'], as_index= False).mean())


# In[327]:


train.head()


# Map Fare according to FareBand.

# In[328]:


for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <=14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) &(dataset['Fare'] <= 31), 'Fare'] = 2  
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[329]:


train.head()

6) SibSp & Parch Feature
# Combining SibSp & Parch feature, we create a new feature named FamilySize.

# In[330]:


for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] +1
    
print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean())


# In[331]:


sns.barplot(x='FamilySize', y='Survived', ci=None, data=train)


# NOTE:
# 
# About data shows that:
#     -Having FamilySize upto 4 (from 2 to 4)  has better survival chance.
#     -FamilySize = 1, i.e. travelling alone has less survival chance.
#     -Large FamilySize (size of 5 and above) also have less survival chance.
Lets create a new feature named IsAlone. This feature isused to check how is the survival chance while travelling alone as compared to travelling with family.
# In[332]:


for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] =1
    
print (train[['IsAlone',"Survived"]].groupby(['IsAlone'], as_index=False).mean())

This shows that travelling alone has only 30% survival chance.
# In[333]:


train.head()


# In[334]:


test.head()

Feature Selection
# We drop unnecessary columns/features and keep only the useful ones for our experiment. Column passengerId is only dropped from Train set because we need PassengerId in Test set while creating Submission file to KAggle

# In[335]:


features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)


# In[336]:


train.head()


# In[337]:


test.head()


# We are done with Feature Selection/Engineering.
# Now, We are ready to train a classifier with our feature set.
Classification  & Accuracy
# Define Training and testing set

# In[338]:


X_train = train.drop('Survived', axis = 1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape


# NOTE:
# There are many classifying algorithms present. Among them, we choose the following Classification algorithms for our problem:
Logistic Regression
Support Vector Machines (SVC)
Linear SVC
k-Nearest Neighbor (KNN)
Decision Tree
Random Forest
Naive Bayes (GaussianNB)
Perceptron
Stochastic Gradient Descent (SGD)
# Here's the training and testing procedure:
First, we train these classifiers with our training data.
After that, using the trained classifier, we predict the Survival outcome of test data.
Finally, we calculate the accuracy score (in percentange) of the trained classifier.
# Please note: that the accuracy score is generated based on our training dataset.

# In[339]:


# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

1) Logistic Regression
# Logistic regression, or logit regression, or logit model is a regression model where the dependent variable (DV) is categorical. This article covers the case of a binary dependent variable—that is, where it can take only two values, "0" and "1", which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. Cases where the dependent variable has more than two outcome categories may be analysed in multinomial logistic regression, or, if the multiple categories are ordered, in ordinal logistic regression.

# In[340]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg  = round(clf.score(X_train, y_train) * 100, 2)
acc_log_reg = round(clf.score(X_train, y_train) *100,2)
print("Train Accuraxy: " +str(acc_log_reg) + '%')

2) Support Vector Machinr (SVM)
# Support Vector Machine (SVM) model is a Supervised Learning model used for classification and regression analysis. It is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
# 
# In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces. Suppose some given data points each belong to one of two classes, and the goal is to decide which class a new data point will be in. In the case of support vector machines, a data point is viewed as a  p -dimensional vector (a list of  p  numbers), and we want to know whether we can separate such points with a  (p−1) -dimensional hyperplane.
# 
# When data are not labeled, supervised learning is not possible, and an unsupervised learning approach is required, which attempts to find natural clustering of the data to groups, and then map new data to these formed groups. The clustering algorithm which provides an improvement to the support vector machines is called support vector clustering and is often used in industrial applications either when data are not labeled or when only some data are labeled as a preprocessing for a classification pass.
# 
# In the below code, SVC stands for Support Vector Classification.

# In[342]:


clf = SVC()
clf.fit(X_train, y_train)

y_pred_svc = clf.predict(X_test)

acc_svc = round(clf.score(X_train, y_train) * 100, 2)

print("Train accuracy: " +str(acc_svc) +'%')

3) Linear SVM
# Linear SVM is a SVM model with linear kernal.
# In the below code, LinearSVC stands for Linear Support Vector Classification

# In[343]:


clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)

print("Train Accuracy: " +str(acc_linear_svc) +'%')

4) K-Nearest Neighbour
# k -nearest neighbors algorithm (k-NN) is one of the simplest machine learning algorithms and is used for classification and regression. In both cases, the input consists of the  k  closest training examples in the feature space. The output depends on whether  k -NN is used for classification or regression:
# 
# In  k -NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its  k  nearest neighbors ( k  is a positive integer, typically small). If  k=1 , then the object is simply assigned to the class of that single nearest neighbor.
# In  k -NN regression, the output is the property value for the object. This value is the average of the values of its  k  nearest neighbors.

# In[344]:


clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) *100, 2)
print("Train Accureacy: " +str(acc_knn) + '%')

5) Decision Tree
# A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). The paths from root to leaf represent classification rules.

# In[345]:


clf =DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) *100, 2)

print("Train Accuracy: " +str(acc_decision_tree) +'%')

6) Random Forest
# Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
# 
# Ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.

# In[350]:


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) *100, 2)

print("Train Accuracy: " +str(acc_random_forest)  +'%')

7) Gaussian Navie Bayes
# Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
# 
# Bayes' theorem (alternatively Bayes' law or Bayes' rule) describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if cancer is related to age, then, using Bayes' theorem, a person's age can be used to more accurately assess the probability that they have cancer, compared to the assessment of the probability of cancer made without knowledge of the person's age.
# 
# Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. It is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.

# In[352]:


clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) *100,2)

print("Train Accuracy: " +str(acc_gnb) +'%')

8) Perceptron
# Perceptron is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector

# In[353]:


clf = Perceptron (max_iter=5, tol=None) 
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train)*100,2)
print("Train Accuracy: " +str(acc_perceptron) +'%')

9) Stochastic Gradient Descent (SGD)
# Stochastic gradient descent (often shortened in SGD), also known as incremental gradient descent, is a stochastic approximation of the gradient descent optimization method for minimizing an objective function that is written as a sum of differentiable functions. In other words, SGD tries to find minima or maxima by iteration.

# In[354]:


clf = SGDClassifier(max_iter= 5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd =clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) *100, 2)

print("Train Accuracy: " +str(acc_sgd) +'%')

Confusion Matrix
# A confusion matrix, also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm. Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabelling one as another).
# 
# In predictive analytics, a table of confusion (sometimes also called a confusion matrix), is a table with two rows and two columns that reports the number of false positives, false negatives, true positives, and true negatives. This allows more detailed analysis than mere proportion of correct classifications (accuracy). Accuracy is not a reliable metric for the real performance of a classifier, because it will yield misleading results if the data set is unbalanced (that is, when the numbers of observations in different classes vary greatly). For example, if there were 95 cats and only 5 dogs in the data set, a particular classifier might classify all the observations as cats. The overall accuracy would be 95%, but in more detail the classifier would have a 100% recognition rate for the cat class but a 0% recognition rate for the dog class.
Here's another guide explaining Confusion Matrix with example
# ActualPositiveActualNegativePredictedPositiveTPFPPredictedNegativeFNTN
In this (Titanic problem) case:
# True Positive: The classifier predicted Survived and the passenger actually Survived.
# True Negative: The classifier predicted Not Survived and the passenger actually Not Survived.
# False Postiive: The classifier predicted Survived but the passenger actually Not Survived.
# False Negative: The classifier predicted Not Survived but the passenger actually Survived.
In the example code below, we plot a confusion matrix for the prediction of Random Forest Classifier on our training dataset. This shows how many entries are correctly and incorrectly predicted by our classifer.
# In[356]:


from sklearn.metrics import confusion_matrix
import itertools


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train)*100,2)
print("Accuracy : %i %% \n"  %acc_random_forest)

class_names = ['Survived', 'Not Survived']

#Compute confusion matrix

cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print('Confusion Matrix in numbers')
print(cnf_matrix)
print('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print('Confusion Matrix in Percentage')
print(cnf_matrix_percent)
print('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']


df_cnf_matrix = pd.DataFrame(cnf_matrix, index = true_class_names, columns = predicted_class_names)


df.cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, index = true_class_names, columns=predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)

sns.heatmap(df_cnf_matrix, annot = True, fmt='d', cmap = "Blues")

Comparing Models
# Let's compare the accuracy score of all the classifier models usesd above

# In[357]:


models = pd.DataFrame({'MOdel': ['LR', 'SVM', 'L-SVC', 'KNN', 'DTree', 'RF', 'NB', 'Perceptron', 'SGD'], 'Score': [acc_log_reg, acc_svc, acc_linear_svc,acc_knn, acc_decision_tree, acc_random_forest, acc_gnb, acc_perceptron, acc_sgd]})

models = models.sort_values(by="Score", ascending=False)

models


# In[359]:


sns.barplot(x='MOdel', y="Score", ci=None, data= models)


# NOTE:
# From the above table, we can see that Decision Tree and Random Forest classfiers have the highest accuracy score.
# Among these two, we choose Random Forest classifier as it has the ability to limit overfitting as compared to Decision Tree classifier.
Create Submission file for Kaggle Completion
# In[360]:


test.head()


# In[361]:


submission = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived": y_pred_random_forest})

submission.to_csv("gender_submission.csv", index = False)


# In[ ]:




