#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

data= pd.read_csv(r"C:\Users\nidhi\OneDrive\Desktop\Heart disease\heart.csv")


# In[3]:


data.head()


# In[4]:


print(data.dtypes)


# In[5]:


data.info


# In[6]:


data.shape


# In[7]:


data.isnull().sum().sort_values(ascending=False)
(data.isnull().mean()*100).sort_values(ascending=False)

#hence, no data is missing


# In[8]:


#Renaming the columns

data.rename(columns={
    'cp': 'chest pain type',
    'trestbps': 'resting blood pressure',
    'chol': 'serum cholestrol in mg/dl',
    'fbs': 'fasting blood sugar>120 mg/dl',
    'restecg': 'resting electrocardiographic results',
    'thalach': 'max heart rate achieved',
    'exang': 'exercise induced angina',
    'oldpeak': 'ST depression',
    'slope': 'the slope of the peak exercise ST segment',
    'ca': 'number of major vessels',
    'thal': ' thalasemia'
}, inplace=True)


# In[9]:


data.sample(10)


# In[10]:


print(data['age'].unique())


# In[11]:


#finding the correlation between the columns

numeric_data = data.drop(columns=['age'])

correlation = numeric_data.corr()['target'].sort_values(ascending=False)

print(correlation)


# In[12]:


#plotting the correlation graph
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation of all features with target
correlation = data.corr()['target'].sort_values(ascending=False)
print(correlation)




# In[13]:


plt.figure(figsize=(6,4))
correlation.drop('target').plot(kind='bar', color='steelblue')
plt.title('Feature Correlation with Target')
plt.axhline(y=0, color='red', linestyle='--')
plt.ylabel('Correlation Score')
plt.show()


# In[14]:


age_map = {
    '20-30': 25,
    '30-40': 35,
    '40-50': 45,
    '50-60': 55,
    '60-70': 65,
    '70-80': 75
}

data['age_group_numeric'] = data['age'].map(age_map)


# In[15]:


data.select_dtypes(include=['number']).shape


# In[16]:


print(data['target'].value_counts())
print(data['exercise induced angina'].value_counts())

# Most important — see the actual relationship
print(data.groupby('exercise induced angina')['target'].mean())


# In[17]:


# Check exang vs target
print("exercise induced angina vs target:")
print(data.groupby('exercise induced angina')['target'].mean())
print()

# Check ca vs target  
print("number of major vessels vs target:")
print(data.groupby('number of major vessels')['target'].mean())
print()

# Check ST depression vs target
print("ST depression vs target:")
print(data.groupby('target')['ST depression'].mean())


# In[18]:


# Flip the target
data['target'] = 1 - data['target']

# Verify after flipping
print(data['target'].value_counts())
print()
print(data.groupby('exercise induced angina')['target'].mean())
print()
print(data.groupby('target')['ST depression'].mean())


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation of all features with target
correlation = data.corr()['target'].sort_values(ascending=False)
print(correlation)



# In[20]:


plt.figure(figsize=(6,4))
correlation.drop('target').plot(kind='bar', color='steelblue')
plt.title('Feature Correlation with Target')
plt.axhline(y=0, color='red', linestyle='--')
plt.ylabel('Correlation Score')
plt.show()


# Observation: Feature Correlation with Target Variable
# During the exploratory data analysis phase, the correlation of each feature with the target variable was computed and visualized as a bar chart.
# Initial Observation (Incorrect Graph):
# Upon plotting the first correlation chart, it was observed that the target variable had been encoded incorrectly — with 1 representing no disease and 0 representing presence of disease — which is the inverse of the standard medical convention. This resulted in correlations that were medically misleading at multiple points:
# 
# ST depression showed a negative correlation with the target, implying it was protective against heart disease — which is clinically incorrect, as ST depression is a well-established marker of myocardial ischemia and elevated cardiac risk.
# Exercise-induced angina also appeared negatively correlated, suggesting it reduced disease likelihood — again, medically unsound.
# Number of major vessels and thalassemia showed inverted relationships, contradicting established cardiology literature.
# Max heart rate achieved appeared positively correlated, implying higher max HR increases risk — whereas clinically, a lower max heart rate during stress testing indicates poorer cardiac reserve and higher risk.
# Chest pain type and slope of the peak exercise ST segment also showed reversed directions inconsistent with their known clinical significance.
# 
# In summary, almost every clinically significant feature had its correlation direction flipped, making the graph medically unreliable.
# Corrective Action (Correct Graph):
# To rectify this, the target variable was re-encoded so that 1 represents presence of heart disease and 0 represents absence, which aligns with standard medical convention. Upon re-plotting the correlation chart, all features showed clinically consistent and medically validated directions:
# 
# ST depression, exercise-induced angina, number of major vessels, and thalassemia correctly showed positive correlations.
# Max heart rate achieved correctly showed a negative correlation.
# The overall pattern of the corrected graph aligns well with established cardiovascular risk literature.

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(4,2))

sns.countplot(data=data, x='sex')

plt.xticks([0, 1], ['Female', 'Male'])
plt.title('Number of Male and Female Patients')
plt.xlabel('Sex')
plt.ylabel('Count')

plt.show()

#OBSERVATION - No. of male patients > No. of female patients


# In[22]:


#creating age groups
import pandas as pd

data['age_group'] = pd.cut(
    data['age'],
    bins=[20, 30, 40, 50, 60, 70, 80],
    labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
)


# In[23]:


#Plotting age group graph
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(4,2))

sns.countplot(
    data=data,
    x='age_group'
)

plt.title('Number of Patients by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')

plt.show()

#OBSERVATION - In the data set- 50-60> 40-50= 60-70> 30-40> 70-80> 20-30


# In[24]:


#finding otliners
import seaborn as sns 

#1. outliner in resting blood pressure
sns.boxplot(y=data['resting blood pressure'])


# In[25]:


#2. Outliner in serum cholestrol mg/dl

sns.boxplot(y=data['serum cholestrol in mg/dl'])


# In[26]:


#3. Outliner in max heart rate achieved

sns.boxplot(y=data['max heart rate achieved'])


# In[27]:


#Graph plots on the basis of sex
import matplotlib.pyplot as plt

sns.barplot(
    data=data,
    x='sex',
    y='resting blood pressure'
)

plt.xticks([0, 1], ['Female', 'Male'])
plt.title('Average Resting Blood Pressure by Sex')
plt.show()


# In[28]:


features = [
    'resting blood pressure',
    'serum cholestrol in mg/dl',
    'max heart rate achieved'
]

for feature in features:
    plt.figure(figsize=(4,2))

    sns.barplot(
        data=data,
        x='sex',
        y=feature
    )

    plt.xticks([0,1], ['Female', 'Male'])
    plt.title(f'{feature} by Sex')
    plt.show()




# 2.KEY INSIGHTS FROM THE SERUM CHOLESTROL in mg/dl by sex Graph
# Normal cholesterol is below 200 mg/dl
# Both males and females in this dataset are above normal on average
# Females are more at risk based on cholesterol levels alone

# In[29]:


#plotting graphs on the basis of sex- fasting blood sugar level

plt.figure(figsize=(4,2))

sns.countplot(
    data=data,
    x='fasting blood sugar>120 mg/dl',
    hue='sex'
)
plt.xticks([0, 1], ['Normal (≤120)', 'High (>120)'])
plt.title('Fasting Blood Sugar by Sex')
plt.show()



# KEY INSIGHTS FROM THE ABOVE GRAPH (Fasting blood sugar by Sex)
# Males dominate the dataset — there are far more male patients than female, so the counts are naturally higher for males in both categories
# Most patients have normal fasting blood sugar (FBS ≤ 120) regardless of sex
# A small proportion have high FBS (>120) in both sexes
# The male-to-female ratio stays roughly similar in both FBS groups, suggesting fasting blood sugar level doesn't strongly differ by sex in this dataset

# In[30]:


plt.figure(figsize=(6,4))

sns.scatterplot(
    data=data,
    x='age',
    y='resting blood pressure',
    hue='sex'
)

plt.title('Age vs Resting Blood Pressure')
plt.show()


# here is a mild positive trend — resting blood pressure tends to increase slightly with age, which is medically expected as arterial stiffness increases over time.
# Sex-wise Observation:
# 
# Both male (orange) and female (blue) patients are fairly evenly distributed across the age and BP range, with no dramatic sex-based separation visible.
# However, some of the higher BP outliers (180–200 mmHg) appear to be female patients, which aligns with the known clinical fact that post-menopausal women often experience sharper BP increases.

# In[44]:


data['age_group'] = pd.cut(
    data['age'],
    bins=[20,30,40,50,60,70,80],
    labels=['20-30','30-40','40-50','50-60','60-70','70-80']
)

sns.countplot(
    data=data,
    x='age_group',
    hue='target'
)

plt.title('Heart Disease by Age Group')
plt.show()


# In[45]:


data = data.drop('age_group', axis=1)


# In[47]:


data = data.drop('age_group_numeric', axis=1)


# In[55]:


#splitting the data into test and train 

from sklearn.model_selection import train_test_split
X = data.drop('target', axis=1)
y= data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")


# In[56]:


#Model Training

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_auc_score, 
                             roc_curve)

models= {
    'LogisticRegression' : LogisticRegression(max_iter=1000, class_weight="balanced"),
    'Decision Tree' : DecisionTreeClassifier (random_state=42, max_depth= 5, min_samples_leaf= 3, max_features= 2),
    'Random forest' : RandomForestClassifier (random_state= 42, n_estimators=50,  bootstrap= True, class_weight="balanced")
}

result= {}

for name, model in models.items():

    #Train
    model.fit(X_train, y_train)

    #Predict
    y_pred= model.predict(X_test)

    result[name] = {
        'model' : model,
        'y_pred' : y_pred,
        'accuracy' : accuracy_score(y_test, y_pred)
    }
    print(f"{name}")
    print(f"Accuracy: {result[name]['accuracy']*100:.2f}%")
    print(classification_report(y_test, y_pred))


# In[57]:


# Add this to verify
rf_model = result['Random forest']['model']
print("Train Accuracy:", accuracy_score(y_train, rf_model.predict(X_train)))
print("Test Accuracy :", accuracy_score(y_test,  rf_model.predict(X_test)))


# In[58]:


from sklearn.model_selection import cross_val_score

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")


# Logistic Regression — 83.41% ✅ Most Reliable
# For Class 0 (No Disease):
# 
# Precision 80% — of all predicted healthy, 80% were actually healthy
# Recall 90% — correctly identified 90% of all actual healthy patients
# 
# For Class 1 (Disease):
# 
# Precision 88% — of all predicted diseased, 88% were actually diseased
# Recall 77% — missed 23% of actual heart disease patients
# 
# Verdict: Most balanced and trustworthy model. The slightly lower recall for disease class (77%) is a concern medically — missing 23% of actual patients is not ideal, but overall the model is stable and consistent.
# 

# Decision Tree — 79.02% ⚠️ Weakest
# For Class 0 (No Disease):
# 
# Precision 82%, Recall 76% — decent but not great
# 
# For Class 1 (Disease):
# 
# Precision 77%, Recall 82% — catches more disease cases but less precise
# 
# Verdict: Lowest accuracy among the three. Results dropped significantly from the single train-test split (83.77%) to cross validation (79.32%), confirming it was previously overfitting. Not recommended as the final model.

# Random Forest — 95.61% ❌ Overfitting
# For Class 0 (No Disease):
# 
# Precision 100%, Recall 91% — perfect precision but suspicious
# 
# For Class 1 (Disease):
# 
# Precision 92%, Recall 100% — catches every disease case but unrealistic
# 
# Verdict: Despite impressive numbers, this model is overfitting. A 100% precision on a 1000-row medical dataset is statistically unrealistic. Combined with the earlier finding of 100% train accuracy, these results cannot be trusted for real-world prediction.
