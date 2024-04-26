#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from collections import Counter
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
# Đọc tệp CSV
loan_df = pd.read_csv(r"C:\Users\hoang\Downloads\dataset - loan application..csv")

print(loan_df.head())


# In[2]:


loan_df.head()


# In[3]:


loan_df.describe()


# In[4]:


len(loan_df[loan_df['Loan_Status']=='N'])


# In[5]:


loan_df.info()


# In[6]:


#1. imputer --> cac thuat toan --> trinh bay
#2. Imbalance --> cac thuat toan, phuong phap xu ly imbalance --> trinh bay
#3. Trinh bay --> EDA --> rut ra dieu j --> anh huong cua cac thuoc tinh toi output
#4. Plan --> giai quyet bai toan (trello)


# In[7]:


sns.set(rc={'figure.figsize':(11.7,8.27)})

# Vẽ các subplot
plt.subplot(231)
sns.countplot(x="Gender", hue='Loan_Status', data=loan_df)
plt.subplot(232)
sns.countplot(x="Married", hue='Loan_Status', data=loan_df)
plt.subplot(233)
sns.countplot(x="Education", hue='Loan_Status', data=loan_df)
plt.subplot(234)
sns.countplot(x="Self_Employed", hue='Loan_Status', data=loan_df)
plt.subplot(235)
sns.countplot(x="Dependents", hue='Loan_Status', data=loan_df)
plt.subplot(236)
sns.countplot(x="Property_Area", hue='Loan_Status', data=loan_df)

# Hiển thị đồ thị
plt.show()


# In[8]:


bins = np.linspace(loan_df.ApplicantIncome.min(), loan_df.ApplicantIncome.max(), 12)
graph = sns.FacetGrid(loan_df, col="Gender", hue="Loan_Status", palette="Set2", col_wrap=2)
graph.map(plt.hist, "ApplicantIncome", bins=bins, ec="k")
graph.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(loan_df.Loan_Amount_Term.min(), loan_df.Loan_Amount_Term.max(), 12)
graph = sns.FacetGrid(loan_df, col="Gender", hue="Loan_Status", palette="Set2", col_wrap=2)
graph.map(plt.hist, "Loan_Amount_Term", bins=bins, ec="k")
graph.axes[-1].legend()

bins = np.linspace(loan_df.CoapplicantIncome.min(), loan_df.CoapplicantIncome.max(), 12)
graph = sns.FacetGrid(loan_df, col="Gender", hue="Loan_Status", palette="Set2", col_wrap=2)
graph.map(plt.hist, "CoapplicantIncome", bins=bins, ec="k")
graph.axes[-1].legend()

plt.show()


# In[10]:


# Loại bỏ các biến phân loại khỏi DataFrame trước khi tính tương quan
numeric_data = loan_df.select_dtypes(include=['number'])

correlation_matrix = numeric_data.corr()

sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, cmap="YlOrRd")
plt.show()


# In[11]:


mask = np.zeros_like(correlation_matrix)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7,6))
    ax = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap="YlOrRd")
plt.show()


# In[12]:


categorical_columns = ['Gender', 'Married', 'Dependents',
                       'Education', 'Self_Employed', 'Property_Area',
                       'Credit_History', 'Loan_Amount_Term']

# Lựa chọn các cột không phải là số
categorical_columns = ['Gender', 'Married', 'Dependents',
                       'Education', 'Self_Employed', 'Property_Area',
                       'Credit_History', 'Loan_Amount_Term']
categorical_df = loan_df[categorical_columns]

# Lựa chọn các cột là số
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'Total_Income']
numerical_df = loan_df[numerical_columns]

loan_df['ApplicantIncome'] = pd.to_numeric(loan_df['ApplicantIncome'], errors='coerce')
loan_df['CoapplicantIncome'] = pd.to_numeric(loan_df['CoapplicantIncome'], errors='coerce')

loan_df['Total_Income'] = loan_df['ApplicantIncome'] + loan_df['CoapplicantIncome']
for feature in numerical_columns:
    plt.figure(figsize=(10, 4))

    # Before log-transformation
    plt.subplot(1, 2, 1) # (row, column, index)
    plt.boxplot(loan_df[feature]) 
    plt.title(f'Before Log-Transformation: {feature}')
    plt.xlabel('Dataset')
    plt.ylabel(feature)

    # After log-transformation
    plt.subplot(1, 2, 2)
    plt.boxplot(np.log1p(loan_df[feature]))  
    plt.title(f'After Log-Transformation: {feature}')
    plt.xlabel('Dataset')
    plt.ylabel(f'Log({feature} + 1)')

    plt.tight_layout()
    plt.show()


# In[13]:


loan_transformed = loan_df.copy()  # Sử dụng df thay vì loan nếu bạn đang sử dụng DataFrame df
loan_copy = loan_df.copy()  # Sử dụng df thay vì loan_copy nếu bạn đang sử dụng DataFrame df

for feature in numerical_columns:
    plt.figure(figsize=(10, 4))

    # Before log-transformation
    plt.subplot(1, 2, 1) # (row, column, index)
    plt.plot(range(len(loan_df[feature])), loan_df[feature].values)
    plt.title(f'Before Log-Transformation: {feature}')
    plt.xlabel('Dataset')
    plt.ylabel(feature)

    # After log-transformation
    plt.subplot(1, 2, 2)
    loan_copy['Total_Income'] = np.log1p(loan_copy['Total_Income'])
    plt.plot(range(len(loan_copy['Total_Income'])), loan_copy['Total_Income'].values)
    plt.title('Log-Transformed Total_Income')
    plt.xlabel('Dataset')
    plt.ylabel('Log(Total_Income + 1)')
    plt.tight_layout()  # Adjust layout to prevent overlapping
    
    plt.show()
    


# In[14]:


missing_values = loan_df.isnull().sum()
print(missing_values)
percentage_missing = (loan_df.isnull().sum() / len(loan_df)) * 100
print(percentage_missing)

# Drop rows with any missing values
df_cleaned = loan_df.dropna()

# Fill missing values with mean
df_filled = loan_df.fillna(loan_df.mean(numeric_only=True),inplace=True)


# In[15]:


for feature in categorical_columns:
  loan_transformed[feature] = np.where(loan_transformed[feature].isnull(),
                                       loan_transformed[feature].mode(),
                                       loan_transformed[feature])
# với những cột giá trị dạng số mà có ô bị thiếu thì sẽ fill bằng giá trị median(aka giá trị xuất hiện ở giữa(khác vs mean đấy))
for feature in numerical_columns:
  loan_transformed[feature] = np.where(loan_transformed[feature].isnull(),
                                       int(loan_transformed[feature].median()),
                                       loan_transformed[feature])


# In[16]:


loan_transformed.isnull().sum()


# In[17]:


target= 'Loan_Status'
loan_transformed[target] = np.where(loan_transformed[target] == 'Y', 1, 0)


# In[18]:


loan_transformed = pd.get_dummies(loan_transformed, drop_first = True)
loan_transformed.head()


# In[19]:


# Sử dụng resample để cân bằng dữ liệu
from sklearn.utils import resample

# Xác định tập dữ liệu thiểu số
minority_class = loan_df['Loan_Status'].value_counts().sort_values().index[0]

# Oversample tập dữ liệu thiểu số
df_oversampled = resample(loan_df[loan_df['Loan_Status'] == minority_class],
                          replace=True,
                          n_samples=len(loan_df[loan_df['Loan_Status'] != minority_class]))

# Undersample tập dữ liệu đa số
df_undersampled = resample(loan_df[loan_df['Loan_Status'] != minority_class],
                          replace=False,
                          n_samples=len(loan_df[loan_df['Loan_Status'] == minority_class]))

# Kết hợp các tập dữ liệu đã resample
df_resampled = pd.concat([df_oversampled, df_undersampled])


# In[20]:


# Oversample lớp thiểu số
df_oversampled = resample(loan_df[loan_df['Loan_Status'] == 'N'],
                          replace=True,
                          n_samples=len(loan_df[loan_df['Loan_Status'] == 'Y']))

# Kết hợp các tập dữ liệu
df_resampled = pd.concat([df_oversampled, loan_df[loan_df['Loan_Status'] == 'Y']])

# Kiểm tra mức độ cân bằng dữ liệu
print(df_resampled['Loan_Status'].value_counts())


# In[21]:


# Vẽ biểu đồ dữ liệu được cân bằng
sns.countplot(x='Loan_Status', data=loan_transformed)
plt.show()


# In[22]:


# Vẽ biểu đồ dữ liệu được cân bằng
sns.countplot(x='Loan_Status', data=df_resampled)
plt.show()


# In[49]:


#Logistic Regression Model, Decision Tree Classifier, Gradient Boosting, and Random Forest Classifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[50]:


# Load the preprocessed dataset
df = loan_transformed.copy()

# Load the balanced dataset
df_resampled = df_resampled.copy()

# Features
X = df_resampled[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']]

# Target variable
y = df_resampled['Loan_Status']

#Chia ra tập train (75%), tập test (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
numeric_features = ['Credit_History', 'Loan_Amount_Term']


# In[51]:


print("Training X data:")
print(X_train.head())

print("\nTesting X data:")
print(X_test.head())

print("\nTesting y data:")
print(y_train.head())

print("\nTesting y data:")
print(y_test.head())


# In[69]:


# One-hot encode categorical features and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)

# Define the logistic regression model within the pipeline with regularization, định nghĩa model logistic regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, penalty='l2', solver='lbfgs'))
])

# Fit the model on the training data
model.fit(X_train, y_train)


# In[70]:


# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy of logistic regression
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of logistic regression:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[73]:


# Initialize the decision tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Create pipeline
model2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', decision_tree_model)
])

# Fit the model to the training data
model2.fit(X_train, y_train)


# In[74]:


# Predict on the test set
y_pred_decision_tree = model2.predict(X_test)

# Calculate accuracy
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print("Decision Tree Model Accuracy:", accuracy_decision_tree)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_decision_tree))


# In[75]:


# Initialize the Gradient Boosting model
gradient_boosting_model = GradientBoostingClassifier(random_state=42)

# Create pipeline
model3 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gradient_boosting_model)
])

# Fit the model to the training data
model3.fit(X_train, y_train)


# In[76]:


# Predict on the test set
y_pred_gradient_boosting = model3.predict(X_test)

# Calculate accuracy
accuracy_gradient_boosting = accuracy_score(y_test, y_pred_gradient_boosting)
print("Gradient Boosting Model Accuracy:", accuracy_gradient_boosting)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_gradient_boosting))


# In[77]:


# Initialize the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)

# Create pipeline
model4 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', random_forest_model)
])

# Fit the model to the training data
model4.fit(X_train, y_train)


# In[78]:


# Predict on the test set
y_pred_model4 = model4.predict(X_test)

# Calculate accuracy
accuracy_model4 = accuracy_score(y_test, y_pred_model4)
print("Random Forest Model Accuracy:", accuracy_model4)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_model4))


# In[58]:


# Predict loan status based on user input aftering choosing the best model: Random Forest (with the highest accuracy 0.79)
def predict_loan_status_model4(model):
    print("Please enter the following information:")
    gender = input("Gender (Male/Female): ").capitalize()
    married = input("Married (Yes/No): ").capitalize()
    dependents = input("Dependents: ")
    education = input("Education (Graduate/Not Graduate): ").capitalize()
    self_employed = input("Self Employed (Yes/No): ").capitalize()
    property_area = input("Property Area (Urban/Rural/Semiurban): ").capitalize()
    credit_history = float(input("Credit History (1.0/0.0): "))
    loan_amount_term = float(input("Loan Amount Term: "))

    user_data = pd.DataFrame({'Gender': [gender], 
                              'Married': [married], 
                              'Dependents': [dependents], 
                              'Education': [education], 
                              'Self_Employed': [self_employed], 
                              'Property_Area': [property_area], 
                              'Credit_History': [credit_history], 
                              'Loan_Amount_Term': [loan_amount_term]})

    predicted_result = model.predict(user_data)
    print("\nPredicted Loan Status (Random Forest):", predicted_result[0])

# Predict loan status based on user input using Random Forest model
predict_loan_status_model4(model4)

