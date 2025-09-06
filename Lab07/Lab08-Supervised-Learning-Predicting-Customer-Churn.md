# Lab 08 Supervised Learning: Predicting Customer Churn

## **Mục tiêu học tập**
Sau khi hoàn thành bài học này, học viên sẽ có khả năng:
1. Thực hiện các tác vụ phân loại sử dụng logistic regression
2. Triển khai pipeline phân tích dữ liệu OSEMN (Obtain, Scrub, Explore, Model, iNterpret)
3. Phân tích mối quan hệ giữa biến target và biến giải thích thông qua data exploration
4. Lựa chọn features hiệu quả cho mô hình dự đoán
5. Xây dựng mô hình churn prediction sử dụng logistic regression

---

## **Bài tập Thực hành**
### Bài tập cơ bản

#### **Exercise 7.01: Comparing Predictions by Linear and Logistic Regression on the Shill Bidding Dataset**
Consider the **Shill_Bidding_Dataset.csv** dataset, which contains details regarding auctions done for various products on eBay.com. The target column, **Class**, provides information about the bidding behavior, **0** being normal and **1** being abnormal behavior. Abnormal behavior can be similar to malicious clicks or automatic bidding. You have been asked to develop a machine learning model that can predict whether the bidding behavior in a particular auction is normal (**0**) or not (**1**). Apply linear and logistic regression to predict the output and check which one of them is useful in this situation:


_Perform the following steps to achieve the aim of the exercise:_

**Code:**

```python
# 1.	Import the pandas, numpy, sklearn, and matplotlib libraries using the following code:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

# 2.	Create a new DataFrame and name it data. Look at the first few rows using the following code:
data = pd.read_csv("Shill_Bidding_Dataset.csv")
data.head()

# 3.	Next, remove the columns that are irrelevant to the case study; that is, remove the Record_ID, Auction_ID,
# 		and Bidder_ID columns. This is because these columns contain unique IDs and thus do not add any new information
# 		to the model:
data.drop(["Record_ID","Auction_ID","Bidder_ID"],axis=1, inplace=True)

# 4.	Now view the first five rows of the revised data:
data.head()

# 5.	Split the data into training and testing sets as follows. We are sticking to the default value for test data 
# 		size (30% of the entire data). Moreover, to add reproducibility, we are using a random_state of 1, and to account  
# 		for any class imbalance, we will use stratified splitting. Reproducibility will ensure that when someone else  
# 		runs the code, they will get similar results. Stratified splitting will ensure that we take into account 
# 		the class distribution of the target column while splitting the data into train and test sets:
X = data.drop("Class",axis=1)
y = data["Class"]

# Split the dataset into training and testing sets 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=1, stratify=y)

# 6.	Print the size of training and testing datasets:
print("Training dataset size: {}, Testing dataset size: {}".format(X_train.shape,X_test.shape))

# 7.	Fit the model using linear regression:
linear = linear_model.LinearRegression()
linear.fit(X_train,y_train)


# 8.	Predict on the first 10 test data points using the following code:
linear.predict(X_test)[:10]

# 9.	Now check the actual target values using y_test[:10].values:
y_test[:10].values

# 10.	Evaluate the score of the linear regression model on the training and testing datasets:
print("Score on training dataset: {}, "\
       "Score on testing dataset: {}"\
       .format(linear.score(X_train,y_train), linear.score(X_test,y_test)))

# 11.	Fit the model using logistic regression as follows:
logit = linear_model.LogisticRegression()
logit.fit(X_train,y_train)

# 12.	Predict on the test data:
logit.predict(X_test)[:10]

# 13.	Check the actual target values with y_test[:10].values:
y_test[:10].values

# 14.	Similar to linear regression, find the score on the training and testing datasets as follows:
print("Score on training dataset: {}, "\
       "Score on testing dataset: {}"\
       .format(logit.score(X_train,y_train), logit.score(X_test,y_test)))

```


#### **Exercise 7.02: Obtaining the Data**
In this exercise, you will import the banking data (Churn_Modelling.csv) provided by the bank and do some initial checks, such as seeing how many rows and columns are present. This will give you a quick peek into the real-life problem statements that a marketing analyst and a data scientist will work on, where the dataset is not always clean. The more time you spend improving and getting familiar with the dataset, the better the observations you can make about the trend will be:

**Code:**

```python
# 1.	Import the pandas, numpy, matplotlib, and seaborn libraries:
import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns

# 2.	Read the data into a pandas DataFrame named data. Also, view the first five rows of the dataset:
data= pd.read_csv('Churn_Modelling.csv') data.head(5)

# 3. Check the number of rows and columns in the dataset:
len(data) data.shape

```

---

#### **Exercise 7.03: Imputing Missing Values**
After reading the banking data, you have to find any missing values in the data and perform imputation on the missing values:

**Code:**

```python
# 1.	Check for any missing values first using the following code:
data.isnull().values.any()

# 2.	Explore the columns that have these missing values by using the following code:
data.isnull().any()

# 3.	Use describe function to explore the data in the Age and EstimatedSalary columns:
data[["EstimatedSalary","Age"]].describe()

# 4.	Now use describe function for the entire DataFrame as well.
# 		This will help in understanding the statistical description of the columns:
data.describe()

# 5.	Now, check the count of 0s and 1s in this column using the following syntax:
data['HasCrCard'].value_counts()

# 6.	Use the following syntax to find out the total number of missing values:
data.isnull().sum()

# 7.	Find out the percentage of missing values using the following code:
round(data.isnull().sum()/len(data)*100,2)

# 8. Check the data types of the missing columns:
data[["Gender","Age","EstimatedSalary"]].dtypes

# 9.	Now you need to impute the missing values. You can do that by dropping the rows that have missing values,
# 		filling in the missing values with a test statistic (such as mean, mode, or median), or predicting
# 		the missing values using a machine learning algorithm. For EstimatedSalary,
# 		fill in the missing values with the mean of the data in that column using the following code:
mean_value=data['EstimatedSalary'].mean()
data['EstimatedSalary']=data['EstimatedSalary'].fillna(mean_value)

# 10.	For Gender, use value_count() to see how many instances of each gender are present:
data['Gender'].value_counts()

data['Gender']=data['Gender'].fillna(data['Gender'].value_counts().idxmax())

# 11.	For Age, use mode() to get the mode of the data, which is 37, and then replace the missing values with
# 		the mode of the values in the column using the following code:
data['Age'].mode() mode_value=data['Age'].mode()
data['Age']=data['Age'].fillna(mode_value[0])

# 12.	Check whether the missing values have been imputed:
data.isnull().any()

```


---

#### **Exercise 7.04: Renaming Columns and Changing the Data Type**
Scrubbing data also involves renaming columns in the right format and can include removing any special characters and spaces in the column names, shifting the target variable either to the extreme left or right for better visibility, and checking whether the data types of the columns are correct. In this exercise, you have to convert the column names into a more human-readable format. For example, you must have noticed column names such as ActMem and CredRate. These can be renamed to give a clearer idea of what the columns represent. The main reason behind doing this is that if someone else is going over your work, ambiguous column names can reduce the clarity. Therefore, in this exercise, you will rename some of the columns, change the data types, and shift the Churn column to the rightmost position. This will help differentiate the independent features from the dependent ones:


**Code:**

```python
# 1.	Rename the CredRate, ActMem, Prod Number, and Exited columns using the following command:
data = data.rename(columns={'CredRate': 'CreditScore', 'ActMem' : 'IsActiveMember',\
                            'Prod Number': 'NumOfProducts', 'Exited':'Churn'}) 

# 2.	Check that the preceding columns have been appropriately renamed using the columns command:
data.columns

# 3.	Move the Churn column to the right and drop the CustomerId column using the following code.
# 		You will need to drop the CustomerId column since it is unique for each entry and thus,
# 		does not provide any useful information:
data.drop(labels=['CustomerId'], axis=1,inplace = True)
column_churn = data['Churn']
data.drop(labels=['Churn'], axis=1,inplace = True)
data.insert(len(data.columns), 'Churn', column_churn.values)

# 4.	Check whether the order of the columns has been fixed using the following code:
data.columns

# 5.	Change the data type of the Geography, Gender, HasCrCard, Churn, and IsActiveMember columns to category as 
# 		shown. Recall that these columns were initially strings or objects. However, since these are distinct values,
# 		you will need to convert them to categorical variables by converting the data type to category:
data["Geography"] = data["Geography"].astype('category')
data["Gender"] = data["Gender"].astype('category')
data["HasCrCard"] = data["HasCrCard"].astype('category')
data["Churn"] = data["Churn"].astype('category')
data["IsActiveMember"] = data["IsActiveMember"].astype('category')

# 6.	Now check whether the data types have been converted or not using the following code:
data.dtypes

```


---

#### **Exercise 7.05: Obtaining the Statistical Overview and Correlation Plot**
You are requested to find out the number of customers that churned using basic exploration techniques. The churn column has two attributes: 0 indicates that the customer did not churn and 1 implies that the customer churned. You will be required to obtain the percentage of customers who churned, the percentage of customers who have a credit card, and more. This information will be valuable at a later stage to make inferences about consumer behavior. You are also required to plot the correlation matrix, which will give you a basic understanding of the relationship between the target variable and the rest of the variables.

**Code:**

```python
# 1.	Inspect the target variable to see how many of the customers have churned using the following code:
data['Churn'].value_counts(0)

# 2.	Inspect the percentage of customers who left the bank using the following code:
data['Churn'].value_counts(1)*100

# 3.	Inspect the percentage of customers that have a credit card using the following code:
data['HasCrCard'].value_counts(1)*100

# 4.	Get a statistical overview of the data:
data.describe()


# 5.	Inspect the mean attributes of customers who churned compared to those who did not churn:
summary_churn = data.groupby('Churn') summary_churn.mean()


# 6. Also, find the median attributes of the customers:
summary_churn.median()

# 7. Now use the seaborn library to plot the correlation plot using the following code:
corr = data.corr()
plt.figure(figsize=(15,8)) sns.heatmap(corr, xticklabels=corr.columns.values,\
             yticklabels=corr.columns.values, annot=True,cmap='Greys_r')
corr

```


---

#### **Exercise 7.06: Performing Exploratory Data Analysis (EDA)**
In this exercise, you will perform EDA, which includes univariate analysis and bivariate analysis, on the Churn_Modelling.csv dataset. You will use this analysis to come up with inferences regarding the dataset, and the relationship between features such as geography, customer age, customer bank balance, and more with respect to churn:


**Code:**

```python
# 1.	Start with univariate analysis. Plot the distribution graph of the customers for
# 		the EstimatedSalary, Age, and Balance variables using the following code:
f, axes = plt.subplots(ncols=3, figsize=(15, 6)) 
sns.distplot(data.EstimatedSalary, kde=True, color="gray",  ax=axes[0]).set_title('EstimatedSalary')
axes[0].set_ylabel('No of Customers')

sns.distplot(data.Age, kde=True, color="gray",  ax=axes[1]).set_title('Age')
axes[1].set_ylabel('No of Customers') 

sns.distplot(data.Balance, kde=True, color="gray", 
ax=axes[2]).set_title('Balance')
axes[2].set_ylabel('No of Customers')

# 2.	Now, move on to bivariate analysis. Inspect whether there is a difference in churn for
# 		Gender using bivariate analysis. Use the following code:
plt.figure(figsize=(15,4))
p=sns.countplot(y="Gender", hue='Churn', data=data, palette="Greys_r")

legend = p.get_legend()
legend_txt = legend.texts
legend_txt[0].set_text("No Churn")
legend_txt[1].set_text("Churn") 
p.set_title('Customer Churn Distribution by Gender')

# 3. Plot Geography versus Churn:
plt.figure(figsize=(15,4))
p=sns.countplot(x='Geography', hue='Churn', data=data, palette="Greys_r")

legend = p.get_legend()
legend_txt = legend.texts
legend_txt[0].set_text("No Churn")
legend_txt[1].set_text("Churn") 
p.set_title('Customer Geography Distribution')

# 4. Plot NumOfProducts versus Churn:
plt.figure(figsize=(15,4))
p=sns.countplot(x='NumOfProducts', hue='Churn', data=data, palette="Greys_r")

legend = p.get_legend()
legend_txt = legend.texts
legend_txt[0].set_text("No Churn")
legend_txt[1].set_text("Churn")
p.set_title('Customer Distribution by Product')

# 5. Inspect Churn versus Age:
plt.figure(figsize=(15,4))
ax=sns.kdeplot(data.loc[(data['Churn'] == 0),'Age'] , color=sns.color_palette("Greys_r")[0], shade=True,label='no churn', linestyle='--')
ax=sns.kdeplot(data.loc[(data['Churn'] == 1),'Age'] , color=sns.color_palette("Greys_r")[1], shade=True, label='churn')
ax.set(xlabel='Customer Age', ylabel='Frequency')
plt.title('Customer Age - churn vs no churn')
plt.legend()

# 6. Plot Balance versus Churn:
plt.figure(figsize=(15,4))
ax=sns.kdeplot(data.loc[(data['Churn'] == 0),'Balance'] , color=sns.color_palette("Greys_r")[0], shade=True,label='no churn',linestyle='--')
ax=sns.kdeplot(data.loc[(data['Churn'] == 1),'Balance'] , color=sns.color_palette("Greys_r")[1], shade=True, label='churn')
ax.set(xlabel='Customer Balance', ylabel='Frequency')

plt.title('Customer Balance - churn vs no churn')
plt.legend()

# 7. Plot CreditScore versus Churn:
plt.figure(figsize=(15,4))
ax=sns.kdeplot(data.loc[(data['Churn'] == 0),'CreditScore'] , color=sns.color_palette("Greys_r")[0], shade=True,label='no churn',linestyle='--')
ax=sns.kdeplot(data.loc[(data['Churn'] == 1),'CreditScore'] ,  color=sns.color_palette("Greys_r")[1], shade=True, label='churn')
ax.set(xlabel='CreditScore', ylabel='Frequency')

plt.title('Customer CreditScore - churn vs no churn')
plt.legend()

```

---

#### **Activity 7.01: Performing the OSE technique from OSEMN**
A large telecom company wants to know why customers are churning. You are tasked with first finding out the reason behind the customer churn and then preparing a plan to reduce it. For this purpose, you have been provided with some data regarding the current bill amounts of customers (Current Bill Amt), the average number of calls made by each customer (Avg Calls), the average number of calls made by customers during weekdays (Avg Calls Weekdays), how long each account has been active (Account Age), and the average number of days the customer has defaulted on their bill payments (Avg Days Delinquent). To solve the first problem, you will use the OSE technique from OSEMN to carry out an initial exploration of the data.
_ Follow these steps:_

1.	Import the necessary libraries.

2.	Download the dataset and save it in a file called **Telco_Churn_Data.csv**.

3.	Read the Telco_Churn_Data.csv dataset and look at the first few rows of the dataset. You should get the following output:

![Figure 7.43: The first few rows of read.csv](images/Figure-7.43.jpg)

4.	Check the length and shape of the data (the number of rows and columns). The length should be 4708 and the shape should be (4708, 15).

5.	Rename all the columns in a readable format. Make the column names look consistent by separating them with _ instead of spaces, for example, rename 
Target Code to Target_Code. Also, fix the typo in the  
Avg_Hours_WorkOrderOpenned column. Your column names should finally look as follows.

![Figure 7.44: Renamed column names](images/Figure-7.44.jpg)

6.	Change the data type of the Target_Code,  
Condition_of_Current_Handset, and Current_TechSupComplaints columns from continuous to the categorical object type.

7.	Check for any missing values.

8.	Perform data exploration by initially exploring the Target_Churn variable. You should get the following summary:

![Figure 7.45: Summary of Target_Churn](images/Figure-7.45.jpg)

9.	Find the correlation among different variables and explain the results. You should get the following statistics:

![Figure 7.46: Correlation statistics of the variables](images/Figure-7.46.jpg)

You should get the following plot:

![Figure 7.47: Correlation plot of different features](images/Figure-7.47.jpg)

10.	Perform univariate and bivariate analyses.
For univariate analysis, use the following columns: Avg_Calls_Weekdays, Avg_Calls, and Current_Bill_Amt. You should get the following plots:

![Figure 7.48: Univariate analysis](images/Figure-7.48.jpg)

For bivariate analysis, you should get the following plots.
First, the plot of Complaint_Code versus Target_Churn:

![Figure 7.49: Customer complaint code distribution by churn](images/Figure-7.49.jpg)

Then, the plot of Acct_Plan_Subtype versus Target_Churn:
	
![Figure 7.50: Customer account plan subtype distribution by churn](images/Figure-7.50.jpg)  

Then, the plot of Current_TechSupComplaints versus Target_Churn:

![Figure 7.51: Customer technical support complaints distribution by churn](images/Figure-7.51.jpg)  

Next, the plot of Avg_Days_Delinquent versus Target_Code:

![Figure 7.52: Distribution of the average number of days delinquent by churn](images/Figure-7.52.jpg)  

Then, the plot of Account_Age versus Target_Code:

![Figure 7.53: Distribution of account age by churn](images/Figure-7.53.jpg)  

Lastly, the plot of Percent_Increase_MOM versus Target_Code:

![Figure 7.54: Distribution of the percentage increase of month-on-month  usage by churn/no ](images/Figure-7.54.jpg)  


**Code:**

```python

```

---

#### **Exercise 7.07: Performing Feature Selection**
In this exercise, you will be performing feature selection using a tree-based selection method that performs well on classification tasks. By the end of this exercise, you will be able to extract the most relevant features that can then be used for model building. 
You will be using a different kind of classifier called random forest in this exercise. While we will go into the details of this in the next chapter, the intention is to show how to perform feature selection using a given model. The process for using the random forest classifier is the same as logistic regression using scikit-learn, with the only difference being that instead of importing linear_model, you will need to use the sklearn.ensemble package to import RandomForestClassifier. The steps given in this exercise will provide more details about this.


**Code:**

```python
# 1.	Import RandomForestClassifier and train_test_split from the sklearn library:
from sklearn.ensemble import RandomForestClassifier from sklearn.model_selection import train_test_split

# 2.	Encode the categorical variable using the following code:
data.dtypes ### Encoding the categorical variables data["Geography"] = data["Geography"].astype('category')\                     .cat.codes data["Gender"] = data["Gender"].astype('category').cat.codes data["HasCrCard"] = data["HasCrCard"].astype('category')\                     .cat.codes data["Churn"] = data["Churn"].astype('category').cat.codes

# 3.	Split the data into training and testing sets as follows:
target = 'Churn' X = data.drop('Churn', axis=1) y=data[target]
X_train, X_test, y_train, y_test = train_test_split\                                    (X,y,test_size=0.15, \                                     random_state=123, \                                     stratify=y)

# 4.	Fit the model using the random forest classifier for feature selection with the following code:
forest=RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train,y_train)

# 5.	Call the random forest feature_importances_ attribute to find the important features and store them in a variable named importances:
importances=forest.feature_importances_

# 6.	Create a variable named features to store all the columns, except the target Churn variable. Sort the important features present in the importances variable using NumPy's argsort function:
features = data.drop(['Churn'],axis=1).columns
indices = np.argsort(importances)[::-1]

# 7.	Plot the important features obtained from the random forest using Matplotlib's plt attribute:
plt.figure(figsize=(15,4)) plt.title("Feature importances using Random Forest") plt.bar(range(X_train.shape[1]), importances[indices],\         color="gray", align="center") plt.xticks(range(X_train.shape[1]), features[indices], \            rotation='vertical',fontsize=15) plt.xlim([-1, X_train.shape[1]]) plt.show()

# 8. Place the features and their importance in a pandas DataFrame using the following code:
feature_importance_df = pd.DataFrame({"Feature":features,\                                       "Importance":importances}) print(feature_importance_df)

```
---

#### **Exercise 7.08: Building a Logistic Regression Model**

In the previous exercise, you extracted the importance values of all the features. Next, you are asked to build a logistic regression model using the five most relevant features for predicting the churning of a customer. The customer's attributes are as follows:
•	Age: 50
•	EstimatedSalary: 100,000
•	CreditScore: 600
•	Balance: 100,000
•	NumOfProducts: 2
Logistic regression has been chosen as the base model for churn prediction because of its easy interpretability. 



**Code:**

```python
# 1.	Import the statsmodel package and select only the top five features that you got from the previous exercise to fit your model. Use the following code:
import statsmodels.api as sm 
top5_features = ['Age','EstimatedSalary','CreditScore',\                  'Balance','NumOfProducts'] logReg = sm.Logit(y_train, X_train[top5_features]) logistic_regression = logReg.fit()

# 2.	Once the model has been fitted, obtain the summary and your parameters:
logistic_regression.summary logistic_regression.params

# 3.	Create a function to compute the coefficients. This function will first multiply each feature by its coefficient (obtained in the previous step) and then finally add up the values for all the features in order to compute the final target value:
coef = logistic_regression.params
def y (coef, Age, EstimatedSalary, CreditScore, Balance, \        NumOfProducts) : return coef[0]*Age+ coef[1]\
                        *EstimatedSalary+coef[2]*CreditScore\
                        +coef[1]*Balance+coef[2]*NumOfProducts

# 4.	Calculate the chance of a customer churning by inputting the following values: 
Age: 50
EstimatedSalary: 100,000
CreditScore: 600
Balance: 100,000
NumOfProducts: 2
Use the following code (here, we are implementing the formula we saw in Figure 7.4):
import numpy as np y1 = y(coef, 50, 100000, 600,100000,2) p = np.exp(y1) / (1+np.exp(y1)) p

# 5.	In the previous steps, you learned how to use the statsmodel package. In this step, you will implement scikit-learn's LogisticRegression module to build your classifier and predict on the test data to find out the accuracy of our model:
from sklearn.linear_model import LogisticRegression

# 6.	Fit the logistic regression model on the partitioned training data that was prepared previously:
clf = LogisticRegression(random_state=0, solver='lbfgs')\
      .fit(X_train[top5_features], y_train)

# 7.	Call the predict and predict_proba functions on the test data:
clf.predict(X_test[top5_features]) clf.predict_proba(X_test[top5_features])
<img width="468" height="189" alt="image" src="https://github.com/user-attachments/assets/2b4610a3-0964-4e2f-946a-7dcf194983f0" />

# 8. Calculate the accuracy of the model by calling the score function:
clf.score(X_test[top5_features], y_test)

```

---

#### **Activity 7.02: Performing the MN technique from OSEMN**
You are working as a data scientist for a large telecom company. The marketing team wants to know the reasons behind customer churn. Using this information, they want to prepare a plan to reduce customer churn. Your task is to analyze the reasons behind the customer churn and present your findings.
After you have reported your initial findings to the marketing team, they want you to build a machine learning model that can predict customer churn. With your results, the marketing team can send out discount coupons to customers who might otherwise churn. Use the MN technique from OSEMN to construct your model.


1.	Import the necessary libraries.

2.	Encode the Acct_Plan_Subtype and Complaint_Code columns using the the.astype('category').cat.codes command.

3.	Split the data into training (80%) and testing sets (20%).

4.	Perform feature selection using the random forest classifier. You should get the following output:

![Figure 7.60: Feature importance using random forest](images/Figure-7.60.jpg)

5.	Select the top seven features and save them in a variable named  top7_features. 

6.	Fit a logistic regression using the statsmodel package.

7.	Find out the probability that a customer will churn when the following data is used: Avg_Days_Delinquent: 40, Percent_Increase_MOM: 5,  Avg_Calls_Weekdays: 39000, Current_Bill_Amt: 12000,  Avg_Calls: 9000, Complaint_Code: 0, and Account_Age: 17.
The given customer should have a value of around 81.939% likelihood of churning.

---
## Bài tập tổng hợp

#### Bài tập 1: Model Comparison và Ensemble

**Mục tiêu:** So sánh Logistic Regression với các thuật toán khác và tạo ensemble model

```python
# Hướng dẫn thực hiện:

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def compare_models_exercise():
    """
    Bài tập: So sánh multiple algorithms
    """
    print("="*60)
    print("BÀI TẬP 1: MODEL COMPARISON")
    print("="*60)
    
    # TODO: Implement các models sau
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    # TODO: Train each model và evaluate
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        # TODO: Fit model
        # TODO: Make predictions  
        # TODO: Calculate metrics
        # TODO: Store results
    
    # TODO: Create comparison DataFrame
    # TODO: Visualize results
    # TODO: Identify best performing model
    
    print("Exercise 1 completed!")

# Gợi ý thực hiện:
# 1. Train từng model trên training set
# 2. Evaluate trên test set với multiple metrics
# 3. So sánh AUC-ROC, Precision, Recall, F1
# 4. Analyze trade-offs giữa interpretability và performance
# 5. Create ensemble prediction từ top 3 models
```

#### Bài tập 2: Feature Engineering Advanced

**Mục tiêu:** Tạo thêm features mới và đánh giá impact

```python
def advanced_feature_engineering_exercise():
    """
    Bài tập: Advanced Feature Engineering
    """
    print("="*60)
    print("BÀI TẬP 2: ADVANCED FEATURE ENGINEERING")
    print("="*60)
    
    # TODO: Implement advanced feature engineering techniques
    
    # 1. Polynomial Features
    # TODO: Create polynomial combinations of numerical features
    
    # 2. Interaction Features  
    # TODO: Create interaction terms between important features
    
    # 3. Binning và Discretization
    # TODO: Create optimal bins for continuous variables
    
    # 4. Domain-specific Features
    # TODO: Create business logic based features như:
    #   - Customer loyalty score
    #   - Service utilization ratio
    #   - Payment behavior patterns
    
    # 5. Time-based Features (if available)
    # TODO: Seasonality, trends, recency features
    
    # 6. Feature Selection
    # TODO: Use statistical tests để select best features
    # TODO: Implement recursive feature elimination
    
    # 7. Evaluation
    # TODO: Compare performance before/after feature engineering
    
    print("Exercise 2 completed!")

# Gợi ý:
# - Sử dụng sklearn.preprocessing.PolynomialFeatures
# - Implement custom feature transformations
# - Use feature_selection modules
# - Validate với cross-validation
```

#### Bài tập 3: Threshold Optimization

**Mục tiêu:** Tối ưu decision threshold cho business objectives

```python
def threshold_optimization_exercise():
    """
    Bài tập: Optimize Decision Threshold
    """
    print("="*60)
    print("BÀI TẬP 3: THRESHOLD OPTIMIZATION")
    print("="*60)
    
    # TODO: Implement threshold optimization
    
    # 1. Business Cost Analysis
    # TODO: Define costs of false positives và false negatives
    # Cost of wrongly predicting churn (false positive)
    # Cost of missing actual churn (false negative)
    
    # 2. Threshold Analysis
    # TODO: Test different thresholds từ 0.1 to 0.9
    # TODO: Calculate metrics for each threshold
    
    # 3. ROC Analysis
    # TODO: Find optimal threshold using ROC curve
    # TODO: Calculate Youden's Index
    
    # 4. Precision-Recall Analysis
    # TODO: Find optimal threshold using PR curve
    
    # 5. Business Impact Analysis
    # TODO: Calculate expected profit/loss for each threshold
    # TODO: Find threshold that maximizes business value
    
    # 6. Visualization
    # TODO: Plot metrics vs threshold
    # TODO: Highlight optimal thresholds
    
    print("Exercise 3 completed!")

# Business scenario:
# - Cost of retention campaign: $50 per customer
# - Value of retained customer: $500 (annual value)
# - Find threshold that maximizes ROI
```

#### Bài tập 4: Real-time Prediction System

**Mục tiêu:** Xây dựng system để predict churn cho khách hàng mới

```python
def prediction_system_exercise():
    """
    Bài tập: Build Prediction System
    """
    print("="*60)
    print("BÀI TẬP 4: REAL-TIME PREDICTION SYSTEM")
    print("="*60)
    
    # TODO: Create production-ready prediction system
    
    class ChurnPredictor:
        def __init__(self):
            self.model = None
            self.scaler = None
            self.feature_columns = None
            self.is_trained = False
        
        def train(self, X_train, y_train):
            """Train the model"""
            # TODO: Implement training logic
            pass
        
        def predict_single_customer(self, customer_data):
            """Predict churn for single customer"""
            # TODO: Implement single prediction
            # TODO: Handle data preprocessing
            # TODO: Return probability và risk category
            pass
        
        def predict_batch(self, customers_data):
            """Predict churn for multiple customers"""
            # TODO: Implement batch prediction
            pass
        
        def get_feature_importance(self):
            """Return feature importance explanation"""
            # TODO: Return interpretable feature importance
            pass
        
        def save_model(self, filepath):
            """Save trained model"""
            # TODO: Implement model persistence
            pass
        
        def load_model(self, filepath):
            """Load trained model"""
            # TODO: Implement model loading
            pass
    
    # TODO: Test the system với sample data
    # TODO: Create API endpoint simulation
    # TODO: Implement input validation
    # TODO: Add logging và monitoring
    
    print("Exercise 4 completed!")

# Requirements:
# - Input validation
# - Error handling  
# - Model versioning
# - Performance monitoring
# - Explanation capability
```

#### Bài tập 5: A/B Testing Framework

**Mục tiêu:** Thiết kế A/B test để validate retention strategies

```python
def ab_testing_framework_exercise():
    """
    Bài tập: A/B Testing for Retention Strategies
    """
    print("="*60)
    print("BÀI TẬP 5: A/B TESTING FRAMEWORK")
    print("="*60)
    
    # TODO: Design A/B testing framework
    
    # 1. Experimental Design
    # TODO: Define control và treatment groups
    # TODO: Calculate sample size requirements
    # TODO: Define success metrics
    
    # 2. Customer Segmentation for Testing
    # TODO: Stratify customers by risk level
    # TODO: Ensure balanced groups
    
    # 3. Treatment Strategies
    # TODO: Design different retention interventions:
    #   - Discount offers
    #   - Service upgrades  
    #   - Personalized communications
    #   - Contract incentives
    
    # 4. Statistical Testing
    # TODO: Implement hypothesis testing
    # TODO: Calculate statistical significance
    # TODO: Estimate effect sizes
    
    # 5. Simulation Framework
    # TODO: Simulate different scenarios
    # TODO: Test power analysis
    
    def design_experiment(treatment_effect=0.05, alpha=0.05, power=0.8):
        """Design A/B test experiment"""
        # TODO: Calculate required sample size
        # TODO: Define randomization strategy
        # TODO: Set up tracking metrics
        pass
    
    def analyze_results(control_group, treatment_group):
        """Analyze A/B test results"""
        # TODO: Statistical significance testing
        # TODO: Confidence intervals
        # TODO: Business impact estimation
        pass
    
    print("Exercise 5 completed!")

# Scenario:
# Test effectiveness of different retention campaigns
# Target: Reduce churn by 20% for high-risk customers
# Budget: $10,000 for testing
# Timeline: 3 months
```

---
## Hướng dẫn chấm điểm và đánh giá

### Rubric cho các bài tập:

**Bài Tập 1 (Cơ bản)**:
- Code functionality (40%)
- Data analysis quality (30%) 
- Business insights (20%)
- Presentation (10%)

**Bài Tập 2 (Trung bình)**:
- Technical implementation (35%)
- RFM analysis depth (25%)
- Method comparison (20%) 
- Strategic recommendations (20%)

**Bài Tập 3 (Nâng cao)**:
- Advanced techniques usage (30%)
- Feature engineering creativity (25%)
- Business impact analysis (25%)
- Innovation and insights (20%)

### Tiêu chí đánh giá chung:
- **Xuất sắc (90-100%)**: Vượt expectation, có insights độc đáo
- **Tốt (80-89%)**: Hoàn thành tốt tất cả requirements  
- **Khá (70-79%)**: Hoàn thành cơ bản với một số thiếu sót
- **Trung bình (60-69%)**: Hoàn thành một phần, thiếu insights
- **Yếu (<60%)
